# Video Diffusion Models Experiments

Here we push the boundaries of video generation using diffusion models. This repo will serve as a collection of techniques, including the use of IP Adapters and other methodologies for generating high-quality videos from models traditionally trained for image generation but ofcourse not limited to this.

## Using IP Adapters for Video Generation

I use IP Adapters to condition a T2I model (normally designed for image generation) and trick it into generating video frames. This is achieved by utilizing denoising steps across a large number of samples, allowing the model to generate a somewhat smooth video when combined with an interpolation algorithm.

| Video generation A | Video generation B | Smoothing and stuff | bicubic interpolation | 
| ------------------ | ------------------ | ------------------- | --------------------- | 
| ![](./assets/rnoised_prev.gif) | ![](./assets/ipsscaled.gif) | ![](./assets/stabilized_video.gif) | ![](./assets/bicubic_smoothing.gif) |

For Inference, `cd ./adapter_interpolation` and run the `inference.py` for further instruction refer to the folder itself !

## Using LTX-MultiFrame inference
I use LTX's new model `ltx-video-2b-v0.9.5`, specially the MultiFrame video generation which is pretty nice as is but I have to get MultiFrame support on huge videos like 10-12 secs of videos with just 2 images which is not really possible with vanilla LTX MultiFrame it just messes it up, so trying out some interpolation for intermidiate frames (does not really work), lets see what else can be done here to make it better ! <br><br>
And Ofcourse the inference has support for 8bit quantization and runs all the required 8Bit Cutlass kernels that makes it probably the fastest OS inference for 8bit LTX also supports TeaCache 

| Video generation A | Video generation B | Interpolation (Failed)  | 2 frame disaster | 
| ------------------ | ------------------ | ------------------- | --------------------- | 
| ![](./assets/video_output_0_a-beautiful-vase-of-glass-with-a_1234_1024x1024x96_0.mp4) | ![](./assets/video_output_0_a-beautiful-vase-of-glass-with-a_134_1024x1024x96_1.mp4) | ![](./assets/video_output_0_a-beautiful-vase-of-glass-with-a_13434_1024x1024x96_0.mp4) | ![](./assets/video_output_0_a-beautifull-girl-transforming_1234_1024x768x96_0-Copy1.mp4) |

For Inference, `cd ./ltx-multiframe` and run the `q8_inference.py` for further instruction refer to the folder itself !

## Future Experiments

**1.** I have some ideas with SVDs, since its just a SD2.1 underneath, there's a lot to explore there

**2.** FreeNoise? I don't really like how FreeNoise works but it is proven that it kinda smooths out the video so i might try it with some of the experiments 

**3.** FIFO-SVD? I really like SVD just because of the simplicity of it, I might try getting FIFO to work with SVD

**4.** diffusion process comes with a lot of issues, specially the latent space, there's been a bunch of models that are now ditching the VAE altogether, I'll surely try that

**5.** migrate to Flow Matching based models, try ip-adapter approach with FLUX, once a real ip-adapter is out for it 

**6.** 16 channel VAEs?  

**7.** MiniMax based training technique for video training 

**8.** Cog based video character generations 

---

