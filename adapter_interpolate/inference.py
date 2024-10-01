from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch 
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection
import numpy as np
from PIL import Image
import os
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from torch.linalg import norm

"""
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
)
"""
pipeline = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V5.0", torch_dtype=torch.float16).to("cuda")
pipeline.scheduler = DDIMScheduler(
  beta_end= 0.012,
  beta_schedule= "scaled_linear",
  beta_start= 0.00085,
  clip_sample= False,
  clip_sample_range= 1.0,
  dynamic_thresholding_ratio= 0.995,
  num_train_timesteps= 1000,
  prediction_type= "epsilon",
  rescale_betas_zero_snr= False,
  sample_max_value= 1.0,
  set_alpha_to_one= False,
  steps_offset= 1,
  thresholding= False,
  timestep_spacing= "leading",
  trained_betas= None,
)
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipeline.set_ip_adapter_scale(1.2)

def prepare_ip_adapter_image_embeds(
    ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
):
    image_embeds = []
    if do_classifier_free_guidance:
        negative_image_embeds = []
    if ip_adapter_image_embeds is None:
        if not isinstance(ip_adapter_image, list):
            ip_adapter_image = [ip_adapter_image]

        if len(ip_adapter_image) != len(pipeline.unet.encoder_hid_proj.image_projection_layers):
            raise ValueError(
                f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(pipeline.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
            )

        for single_ip_adapter_image, image_proj_layer in zip(
            ip_adapter_image, pipeline.unet.encoder_hid_proj.image_projection_layers
        ):
            output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
            single_image_embeds, single_negative_image_embeds = pipeline.encode_image(
                single_ip_adapter_image, device, 1, output_hidden_state
            )

            image_embeds.append(single_image_embeds[None, :])
            if do_classifier_free_guidance:
                negative_image_embeds.append(single_negative_image_embeds[None, :])
    else:
        for single_image_embeds in ip_adapter_image_embeds:
            if do_classifier_free_guidance:
                single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                negative_image_embeds.append(single_negative_image_embeds)
            image_embeds.append(single_image_embeds)

    ip_adapter_image_embeds = []
    for i, single_image_embeds in enumerate(image_embeds):
        single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
        if do_classifier_free_guidance:
            single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
            single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

        single_image_embeds = single_image_embeds.to(device=device)
        ip_adapter_image_embeds.append(single_image_embeds)

    return ip_adapter_image_embeds


def lerp(v1, v2, alpha):
    """
    Perform Linear Interpolation (LERP) between two vectors of size [2, 1, 1280].
    
    :param v1: First tensor of size [2, 1, 1280].
    :param v2: Second tensor of size [2, 1, 1280].
    :param alpha: Interpolation factor (0 <= alpha <= 1).
    :return: Interpolated tensor using LERP.
    """
    return (1 - alpha) * v1 + alpha * v2

def slerpv2(v0: FloatTensor, v1: FloatTensor, t: float|FloatTensor, DOT_THRESHOLD=0.9995):
  '''
  Spherical linear interpolation
  Args:
    v0: Starting vector
    v1: Final vector
    t: Float value between 0.0 and 1.0
    DOT_THRESHOLD: Threshold for considering the two vectors as
                            colinear. Not recommended to alter this.
  Returns:
      Interpolation vector between v0 and v1
  '''
  assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

  # Normalize the vectors to get the directions and angles
  v0_norm: FloatTensor = norm(v0, dim=-1)
  v1_norm: FloatTensor = norm(v1, dim=-1)

  v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(-1)
  v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(-1)

  # Dot product with the normalized vectors
  dot: FloatTensor = (v0_normed * v1_normed).sum(-1)
  dot_mag: FloatTensor = dot.abs()

  # if dp is NaN, it's because the v0 or v1 row was filled with 0s
  # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
  gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
  can_slerp: LongTensor = ~gotta_lerp

  t_batch_dim_count: int = max(0, t.dim()-v0.dim()) if isinstance(t, Tensor) else 0
  t_batch_dims: Size = t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
  out: FloatTensor = zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))

  # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
  if gotta_lerp.any():
    lerped: FloatTensor = lerp(v0, v1, t)

    out: FloatTensor = lerped.where(gotta_lerp.unsqueeze(-1), out)

  # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
  if can_slerp.any():

    # Calculate initial angle between v0 and v1
    theta_0: FloatTensor = dot.arccos().unsqueeze(-1)
    sin_theta_0: FloatTensor = theta_0.sin()
    # Angle at timestep t
    theta_t: FloatTensor = theta_0 * t
    sin_theta_t: FloatTensor = theta_t.sin()
    # Finish the slerp algorithm
    s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
    s1: FloatTensor = sin_theta_t / sin_theta_0
    slerped: FloatTensor = s0 * v0 + s1 * v1

    out: FloatTensor = slerped.where(can_slerp.unsqueeze(-1), out)
  
  return out

def slerp(v1, v2, t):
    """
    Perform Spherical Linear Interpolation (SLERP) between two tensors.

    :param v1: First tensor of shape [2, 1, 1280].
    :param v2: Second tensor of shape [2, 1, 1280].
    :param t: Interpolation factor (0 <= t <= 1).
    :return: Interpolated tensor using SLERP.
    """
    # Normalize the vectors
    v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2_norm = v2 / torch.norm(v2, dim=-1, keepdim=True)

    # Compute the angle (omega) between the vectors
    omega = torch.acos(torch.clamp(torch.bmm(v1_norm, v2_norm.transpose(1, 2)).squeeze(), -1.0, 1.0))
    
    # Compute sine of omega for normalization
    so = torch.sin(omega)

    # If omega is close to zero, return v1 (the two vectors are the same)
    if torch.allclose(so, torch.zeros_like(so), atol=1e-6):
        return v1

    # Perform spherical linear interpolation
    slerp_interp = (torch.sin((1.0 - t) * omega) / so).unsqueeze(1) * v1 + \
                   (torch.sin(t * omega) / so).unsqueeze(1) * v2

    return slerp_interp
def generate_interpolations(v1, v2, num_steps):
    """
    Generate multiple interpolated tensors between two tensors of shape [2, 1, 1280].
    
    :param v1: First tensor of size [2, 1, 1280].
    :param v2: Second tensor of size [2, 1, 1280].
    :param num_steps: Number of intermediate tensors to generate (including the start and end).
    :return: Two lists of interpolated tensors using LERP and SLERP.
    """
    lerp_results = []
    slerp_results = []
    
    for step in range(num_steps):
        alpha = step / (num_steps - 1)  # Alpha ranges from 0 to 1
        
        # Generate interpolated tensors using LERP and SLERP for all pairs
        lerp_interp = lerp(v1, v2, alpha)
        slerp_interp = slerpv2(v1, v2, alpha)
        
        # Append results to the respective lists
        lerp_results.append(lerp_interp)
        slerp_results.append(slerp_interp)
    
    return lerp_results, slerp_results


image1_path = 'test.png'
image2_path = 'no_pag.png'

num_frames = 30  # Number of frames including the original images
output_dir = 'output_frames'

img1 = Image.open(image1_path).convert('RGB')
img2 = Image.open(image2_path).convert('RGB')

img2 = img2.resize(img1.size)

from diffusers.models import ImageProjection

image1 = prepare_ip_adapter_image_embeds(ip_adapter_image=img1, ip_adapter_image_embeds=None, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=True)
image2 = prepare_ip_adapter_image_embeds(ip_adapter_image=img2, ip_adapter_image_embeds=None, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=True)

v1 = image1[0]  
v2 = image2[0][:] 
print(v1.shape)

num_steps = 30  # Generate 30 intermediate tensors

lerp_tensors, slerp_tensors = generate_interpolations(v1, v2, num_steps)

print(f"Generated {len(lerp_tensors)} LERP tensors with shape: {lerp_tensors[0].shape}")
print(f"Generated {len(slerp_tensors)} SLERP tensors with shape: {slerp_tensors[0].shape}")

generator = torch.Generator(device="cpu").manual_seed(6457)
dum=torch.zeros(1,4,1024//pipeline.vae_scale_factor,768//pipeline.vae_scale_factor)
latents=torch.randn_like(dum, device="cuda", dtype=torch.float16)
image_list = []
for i, image in enumerate(slerp_tensors):
    images = pipeline(
        prompt="a girl, high quality, 4K",
        ip_adapter_image=None,
        latents=latents,
        ip_adapter_image_embeds=[image*(i+1/10)],
        negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        num_inference_steps=20,
        generator=generator,
        #height=1024,
        #width=768,
        guidance_scale=5.5,
    ).images[0]
    image_list.append(images)
