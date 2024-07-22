import os,sys

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir,'imagdressing'))

pretrained_dir = os.path.join(now_dir,"pretrained_models")

import torch
import cuda_malloc
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from imagdressing.adapter.resampler import Resampler
from imagdressing.dressing_sd.pipelines.IMAGDressing_v1_pipeline import IMAGDressing_v1
from imagdressing.adapter.attention_processor import CacheAttnProcessor2_0, RefSAttnProcessor2_0, CAttnProcessor2_0


device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"

def resize_img(input_image, max_side=640, min_side=512, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
    w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    return input_image

def load_weights(seed,repo_id="SG161222/Realistic_Vision_V4.0_noVAE"):
    generator = torch.Generator(device=device).manual_seed(seed)
    vae_path = os.path.join(pretrained_dir,"sd-vae-ft-mse")
    # stabilityai/sd-vae-ft-mse
    snapshot_download(repo_id="stabilityai/sd-vae-ft-mse",local_dir=vae_path,allow_patterns=["*.json","*.safetensors"])
    vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=torch.float16, device=device)
    diffusers_path = os.path.join(pretrained_dir,"sdmodel")
    snapshot_download(repo_id=repo_id,local_dir=diffusers_path,ignore_patterns=["Realistic_Vision*","realvis4*"])
    tokenizer = CLIPTokenizer.from_pretrained(diffusers_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(diffusers_path, subfolder="text_encoder").to(
        dtype=torch.float16, device=device)

    image_encoder_path = os.path.join(pretrained_dir,"IP-Adapter")
    snapshot_download(repo_id="h94/IP-Adapter",local_dir=image_encoder_path,allow_patterns=["*.json","*.safetensors"],ignore_patterns=["*sd*"])
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, subfolder="models/image_encoder").to(
        dtype=torch.float16, device=device)
    unet = UNet2DConditionModel.from_pretrained(diffusers_path, subfolder="unet").to(
        dtype=torch.float16,
        device=device)
    
     # load ipa weight
    image_proj = Resampler(
        dim=unet.config.cross_attention_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=16,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )
    image_proj = image_proj.to(dtype=torch.float16, device=device)

    # set attention processor
    attn_procs = {}
    st = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = RefSAttnProcessor2_0(name, hidden_size)
        else:
            attn_procs[name] = CAttnProcessor2_0(name, hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules = adapter_modules.to(dtype=torch.float16, device=device)
    del st

    ref_unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet").to(
        dtype=torch.float16,
        device=device)
    ref_unet.set_attn_processor(
        {name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})  # set cache

    # weights load
    model_ckpt_dir = os.path.join(pretrained_dir,"IMAGDressing")
    snapshot_download(repo_id="feishen29/IMAGDressing",local_dir=model_ckpt_dir,allow_patterns=["*.pt"])
    model_sd = torch.load(os.path.join(model_ckpt_dir,'sd',"IMAGDressing-v1_512.pt"), map_location="cpu")["module"]

    ref_unet_dict = {}
    unet_dict = {}
    image_proj_dict = {}
    adapter_modules_dict = {}
    for k in model_sd.keys():
        if k.startswith("ref_unet"):
            ref_unet_dict[k.replace("ref_unet.", "")] = model_sd[k]
        elif k.startswith("unet"):
            unet_dict[k.replace("unet.", "")] = model_sd[k]
        elif k.startswith("proj"):
            image_proj_dict[k.replace("proj.", "")] = model_sd[k]
        elif k.startswith("adapter_modules"):
            adapter_modules_dict[k.replace("adapter_modules.", "")] = model_sd[k]
        else:
            print(k)

    ref_unet.load_state_dict(ref_unet_dict)
    image_proj.load_state_dict(image_proj_dict)
    adapter_modules.load_state_dict(adapter_modules_dict)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    pipe = IMAGDressing_v1(unet=unet, reference_unet=ref_unet, vae=vae, tokenizer=tokenizer,
                         text_encoder=text_encoder, image_encoder=image_encoder,
                         ImgProj=image_proj,
                         scheduler=noise_scheduler,
                         safety_checker=StableDiffusionSafetyChecker,
                         feature_extractor=CLIPImageProcessor)
    return pipe, generator

class IMAGDressingNode:
    def __init__(self) -> None:
        self.pipe = None
        self.generator = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "cloth":("IMAGE",),
                "seed":("INT",{
                    "default": 42
                })
            },
            "optional":{
                "pose":("IMAGE",),
                "face": ("IMAGR",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_IMAGDressing"

    def generate(self,cloth,seed,pose=None,face=None):
        if self.pipe is None:
            self.pipe, self.generator = load_weights(seed)
        print('====================== pipe load finish ===================')
        num_samples = 1
        clip_image_processor = CLIPImageProcessor()

        img_transform = transforms.Compose([
            transforms.Resize([640, 512], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        if face is None:
            prompt = 'A beautiful woman'
            prompt = prompt + ', best quality, high quality'
            null_prompt = ''
            negative_prompt = 'bare, naked, nude, undressed, monochrome, lowres, bad anatomy, worst quality, low quality'

        cloth = cloth.numpy()[0] * 255
        clothes_img = cloth.astype(np.uint8)
        clothes_img = resize_img(clothes_img)
        vae_clothes = img_transform(clothes_img).unsqueeze(0)
        ref_clip_image = clip_image_processor(images=clothes_img, return_tensors="pt").pixel_values

        output = self.pipe(
            ref_image=vae_clothes,
            prompt=prompt,
            ref_clip_image=ref_clip_image,
            null_prompt=null_prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=640,
            num_images_per_prompt=num_samples,
            guidance_scale=7.5,
            image_scale=1.0,
            generator=self.generator,
            num_inference_steps=50,
        ).images

        out_img = torch.from_numpy(np.array(output[0]) / 255.0).unsequeeze(0)
        print(out_img.shape)
        return (out_img,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "IMAGDressingNode": IMAGDressingNode
}


