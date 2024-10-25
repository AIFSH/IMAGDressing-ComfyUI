import os,sys

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir,'imagdressing'))

pretrained_dir = os.path.join(now_dir,"pretrained_models")

import torch
import cuda_malloc
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download,hf_hub_download
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler,ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from imagdressing.adapter.resampler import Resampler
from imagdressing.dressing_sd.pipelines.IMAGDressing_v1_pipeline import IMAGDressing_v1
from imagdressing.adapter.attention_processor import CacheAttnProcessor2_0, RefSAttnProcessor2_0, CAttnProcessor2_0,LoRAIPAttnProcessor2_0,LoraRefSAttnProcessor2_0

from imagdressing.dressing_sd.pipelines.IMAGDressing_v1_pipeline_controlnet import IMAGDressing_v1 as IMAGDressing_v1_CN
from imagdressing.dressing_sd.pipelines.IMAGDressing_v1_pipeline_ipa_controlnet import IMAGDressing_v1 as IMAGDressing_v1_CN_IPA
from imagdressing.dressing_sd.pipelines.IMAGDressing_v1_pipeline_controlnet_inpainting import IMAGDressing_v1 as IMAGDressing_v1_INP
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from imagdressing.preprocess.utils_mask import get_mask_location
from imagdressing.preprocess.openpose.run_openpose import OpenPose
from imagdressing.preprocess.humanparsing.run_parsing import Parsing


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

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0

    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    # image[image_mask > 0.5] = 0  # set as masked pixel
    # cv.imwrite("control_image.jpg", np.array(image * 255))
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def load_weights(use_case,seed,repo_id="SG161222/Realistic_Vision_V4.0_noVAE"):
    generator = torch.Generator(device=device).manual_seed(seed)
    diffusers_path = os.path.join(pretrained_dir,repo_id.split("/")[-1])
    snapshot_download(repo_id=repo_id,local_dir=diffusers_path,ignore_patterns=["Realistic_Vision*","realvis4*"])
    if use_case != "cartoon":
        vae_path = os.path.join(pretrained_dir,"sd-vae-ft-mse")
        # stabilityai/sd-vae-ft-mse
        snapshot_download(repo_id="stabilityai/sd-vae-ft-mse",local_dir=vae_path,allow_patterns=["*.json","*.safetensors"])
        vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=torch.float16, device=device)
    else:
        vae = AutoencoderKL.from_pretrained(diffusers_path,subfolder="vae").to(dtype=torch.float16, device=device)
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

    ref_unet = UNet2DConditionModel.from_pretrained(diffusers_path, subfolder="unet").to(
        dtype=torch.float16,
        device=device)
    ref_unet.set_attn_processor(
        {name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})  # set cache

    # weights load
    model_ckpt_dir = os.path.join(pretrained_dir,"IMAGDressing")
    snapshot_download(repo_id="feishen29/IMAGDressing",local_dir=model_ckpt_dir,allow_patterns=["*.pt"])
    model_sd = torch.load(os.path.join(model_ckpt_dir,"IMAGDressing-v1_512.pt"), map_location="cpu")["module"]

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

    if use_case in ["base","cartoon"]:
        pipe = IMAGDressing_v1(unet=unet, reference_unet=ref_unet, vae=vae, tokenizer=tokenizer,
                            text_encoder=text_encoder, image_encoder=image_encoder,
                            ImgProj=image_proj,
                            scheduler=noise_scheduler,
                            safety_checker=StableDiffusionSafetyChecker,
                            feature_extractor=CLIPImageProcessor)
   
    if use_case == "controlnet":
        control_net_openpose_path = os.path.join(pretrained_dir,"control_v11p_sd15_openpose")
        snapshot_download("lllyasviel/control_v11p_sd15_openpose",
                          local_dir=control_net_openpose_path,
                          ignore_patterns=["*.bin","*fp16*","*.png"])
        control_net_openpose = ControlNetModel.from_pretrained(control_net_openpose_path,
                                                           torch_dtype=torch.float16).to(device=device)

        pipe = IMAGDressing_v1_CN(vae=vae,reference_unet=ref_unet,unet=unet,
                                  tokenizer=tokenizer,text_encoder=text_encoder,
                                  controlnet=control_net_openpose,image_encoder=image_encoder,
                                  scheduler=noise_scheduler,ImgProj=image_proj,
                                  safety_checker=StableDiffusionSafetyChecker,
                            feature_extractor=CLIPImageProcessor)
    
    if use_case == "inpainting":
        control_net_inpaint_path = os.path.join(pretrained_dir,"control_v11p_sd15_inpaint")
        snapshot_download("lllyasviel/control_v11p_sd15_inpaint",
                          local_dir=control_net_inpaint_path,
                          ignore_patterns=["*.bin","*fp16*","*.png"])
        control_net = ControlNetModel.from_pretrained(control_net_inpaint_path,
                                                  torch_dtype=torch.float16).to(device=device)
        pipe = IMAGDressing_v1_INP(unet=unet, reference_unet=ref_unet, vae=vae, tokenizer=tokenizer,
                            text_encoder=text_encoder, image_encoder=image_encoder,
                            ImgProj=image_proj,
                            scheduler=noise_scheduler,
                            controlnet=control_net,
                            safety_checker=StableDiffusionSafetyChecker,
                            feature_extractor=CLIPImageProcessor)
    return pipe, generator


def load_ipa_weights(repo_id="SG161222/Realistic_Vision_V4.0_noVAE"):
    generator = torch.Generator(device=device).manual_seed(42)
    vae_path = os.path.join(pretrained_dir,"sd-vae-ft-mse")
    # stabilityai/sd-vae-ft-mse
    snapshot_download(repo_id="stabilityai/sd-vae-ft-mse",local_dir=vae_path,allow_patterns=["*.json","*.safetensors"])
    vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=torch.float16, device=device)
    
    diffusers_path = os.path.join(pretrained_dir,repo_id.split("/")[-1])
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
        # lora_rank = hidden_size // 2 # args.lora_rank
        if cross_attention_dim is None:
            attn_procs[name] = LoraRefSAttnProcessor2_0(name, hidden_size)
        else:
            attn_procs[name] = LoRAIPAttnProcessor2_0(hidden_size=hidden_size,
                                                      cross_attention_dim=cross_attention_dim,
                                                      scale=1.0, rank=128,
                                                      num_tokens=4)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules = adapter_modules.to(dtype=torch.float16, device=device)
    del st



    ref_unet = UNet2DConditionModel.from_pretrained(diffusers_path, subfolder="unet").to(
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
        elif k.startswith("adapter_modules") and 'ref' in k:
            adapter_modules_dict[k.replace("adapter_modules.", "")] = model_sd[k]
        else:
            print(k)

    ref_unet.load_state_dict(ref_unet_dict)
    image_proj.load_state_dict(image_proj_dict)
    adapter_modules.load_state_dict(adapter_modules_dict, strict=False)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    control_net_openpose_path = os.path.join(pretrained_dir,"control_v11p_sd15_openpose")
    snapshot_download("lllyasviel/control_v11p_sd15_openpose",
                        local_dir=control_net_openpose_path,
                        ignore_patterns=["*.bin","*fp16*","*.png"])

    control_net_openpose = ControlNetModel.from_pretrained(control_net_openpose_path,
                                                           torch_dtype=torch.float16).to(device=device)
    
    ip_ckpt_path = os.path.join(pretrained_dir,"IP-Adapter","ip-adapter-faceid-plusv2_sd15.bin")
    snapshot_download(repo_id="h94/IP-Adapter-FaceID",local_dir=image_encoder_path,
                        allow_patterns=['*plusv2_sd15.bin'])

    pipe = IMAGDressing_v1_CN_IPA(unet=unet, reference_unet=ref_unet, vae=vae, tokenizer=tokenizer,
                            text_encoder=text_encoder, image_encoder=image_encoder,
                            ip_ckpt=ip_ckpt_path,
                            ImgProj=image_proj, controlnet=control_net_openpose,
                            scheduler=noise_scheduler,
                            safety_checker=StableDiffusionSafetyChecker,
                            feature_extractor=CLIPImageProcessor)
    return pipe, generator



class TextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "encode"

    CATEGORY = "AIFSH_IMAGDressing"

    def encode(self, text):
        return (text, )

class IMAGDressingNode:
    def __init__(self) -> None:
        self.pipe = None
        self.use_case = None
        self.generator = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "cloth":("IMAGE",),
                "prompt":("TEXT",),
                "use_case":(["base","controlnet","ipa_controlnet",
                             "cartoon","inpainting"],),
                "num_inference_steps":("INT",{
                    "default": 50
                }),
                "guidance_scale":("FLOAT",{
                    "default": 7.5
                }),
                "seed":("INT",{
                    "default": 42
                })
            },
            "optional":{
                "pose":("IMAGE",),
                "face": ("IMAGE",),
                "model_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_IMAGDressing"

    def comfyimage2Image(self,comfyimage):
        comfyimage = comfyimage.numpy()[0] * 255
        image_np = comfyimage.astype(np.uint8)
        image = Image.fromarray(image_np)
        return image

    def generate(self,cloth,prompt,use_case,num_inference_steps,guidance_scale,
                 seed,pose=None,face=None,model_image=None):
        if self.use_case != use_case:
            self.use_case = use_case
            repo_id = "SG161222/Realistic_Vision_V4.0_noVAE" if use_case != "cartoon" else "stablediffusionapi/counterfeit-v30"
            if use_case == "ipa_controlnet":
                self.pipe, self.generator = load_ipa_weights()
            else:
                self.pipe, self.generator = load_weights(self.use_case,seed,repo_id=repo_id)
        print('====================== pipe load finish ===================')
        num_samples = 1
        clip_image_processor = CLIPImageProcessor()

        img_transform = transforms.Compose([
            transforms.Resize([640, 512], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        
        prompt = prompt
        prompt = prompt + ', best quality, high quality'
        null_prompt = ''
        negative_prompt = 'bare, naked, nude, undressed, monochrome, lowres, bad anatomy, worst quality, low quality'

        clothes_img = resize_img(self.comfyimage2Image(cloth))
        vae_clothes = img_transform(clothes_img).unsqueeze(0)
        ref_clip_image = clip_image_processor(images=clothes_img, return_tensors="pt").pixel_values

        if self.use_case in ["base","cartoon"]:
            output = self.pipe(
                ref_image=vae_clothes,
                prompt=prompt,
                ref_clip_image=ref_clip_image,
                null_prompt=null_prompt,
                negative_prompt=negative_prompt,
                width=512,
                height=640,
                num_images_per_prompt=num_samples,
                guidance_scale=guidance_scale,
                image_scale=1.0,
                generator=self.generator,
                num_inference_steps=num_inference_steps,
            ).images
        
        if self.use_case in ["controlnet","ipa_controlnet"]:
            pose_image = self.comfyimage2Image(pose)

        if self.use_case == "controlnet":
            output = self.pipe(
                ref_image=vae_clothes,
                prompt=prompt,
                ref_clip_image=ref_clip_image,
                pose_image=pose_image,
                null_prompt=null_prompt,
                negative_prompt=negative_prompt,
                width=512,
                height=640,
                num_images_per_prompt=num_samples,
                guidance_scale=guidance_scale,
                image_scale=1.0,
                generator=self.generator,
                num_inference_steps=num_inference_steps,
            ).images

        if self.use_case == "ipa_controlnet":
            app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))

            face_img = self.comfyimage2Image(face)
            image = cv2.cvtColor(np.asarray(face_img),cv2.COLOR_RGB2BGR)
            faces = app.get(image)

            faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)
            face_clip_image = clip_image_processor(images=face_image, return_tensors="pt").pixel_values

            output = self.pipe(
                ref_image=vae_clothes,
                prompt=prompt,
                ref_clip_image=ref_clip_image,
                pose_image=pose_image,
                face_clip_image=face_clip_image,
                faceid_embeds=faceid_embeds,
                null_prompt=null_prompt,
                negative_prompt=negative_prompt,
                width=512,
                height=640,
                num_images_per_prompt=num_samples,
                guidance_scale=guidance_scale,
                image_scale=0.9,
                ipa_scale=0.9,
                s_lora_scale= 0.2,
                c_lora_scale= 0.2,
                generator=self.generator,
                num_inference_steps=num_inference_steps,
            ).images
        if self.use_case == "inpainting":
            # prepare mask model
            inpainting_path = os.path.join(pretrained_dir,"inpainting")
            hf_hub_download(repo_id="yisol/IDM-VTON",filename="parsing_atr.onnx",subfolder="humanparsing",local_dir=inpainting_path)
            hf_hub_download(repo_id="yisol/IDM-VTON",filename="parsing_lip.onnx",subfolder="humanparsing",local_dir=inpainting_path)
            hf_hub_download(repo_id="yisol/IDM-VTON",filename="body_pose_model.pth",subfolder="openpose/ckpts",local_dir=inpainting_path)
            
            parsing_model = Parsing(0,os.path.join(inpainting_path,"humanparsing"))
            openpose_model = OpenPose(0,os.path.join(inpainting_path,"openpose","ckpts"))
            model_image = self.comfyimage2Image(model_image)
            keypoints = openpose_model(model_image.resize((384, 512)))
            model_parse, _ = parsing_model(model_image.resize((384, 512)))
            mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)

            mask_image = mask.resize((512, 512))
            model_image = model_image.resize((512, 512))
            control_image = make_inpaint_condition(model_image, mask_image)

            output = self.pipe(
                ref_image=vae_clothes,
                prompt=prompt,
                ref_clip_image=ref_clip_image,
                null_prompt=null_prompt,
                negative_prompt=negative_prompt,
                image=model_image,
                mask_image=mask_image,
                control_image=control_image,
                width=512,
                height=640,
                num_images_per_prompt=num_samples,
                guidance_scale=guidance_scale,
                image_scale=1.0,
                generator=self.generator,
                num_inference_steps=num_inference_steps,
            ).images

        out_img = torch.from_numpy(np.array(output[0]) / 255.0).unsqueeze(0)
        print(out_img.shape)
        return (out_img,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "TextNode":TextNode,
    "IMAGDressingNode": IMAGDressingNode
}


