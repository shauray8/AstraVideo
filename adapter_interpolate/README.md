# **Video Diffusion Interpolation Experiments**  

It’s a messy, dive into video frame interpolation with diffusion models. If you’re here expecting polished code and production-ready stuff, **get lost**. This is the playground where frames either interpolate **beautifully** or crash and burn. Enjoy the ride.  

## **What’s in the Box?**  
**Keyframe-based Video Interpolation**  
  > Generate keyframes with **IP-Adapter** and interpolate the rest.

1. **RIFE:** Real-time flow-based interpolation. [RIFE GitHub](https://github.com/hzwer/ECCV2022-RIFE)  

2. **Depth-Aware Video Frame Interpolation**
   [Depth Website](https://sites.google.com/view/wenbobao/dain)  

3. **AMT:** All-Pairs Multi-Field Transforms for Efficient Frame Interpolation
   [AMT GitHub](https://github.com/MCG-NKU/AMT)  

4. **Norm-guided latent space exploration for text-to-image generation**
   [NAO-centroid paper](https://arxiv.org/pdf/2306.08687) <br>
   _might try for frame interpolation or ip-adapter embed interpolation_

## **How It Works (Sort of)**  
- **Keyframes:** Slerp and Lerp for ip-adapter embeds as keyframe interpolation.
- **Chopy-Video**: Keyframes goes into a diffusion model as ip-adapter embeds, which is merged to form a video.
- **Video-Interpolation:** RIFE, Depth Stuff, AMT... take your pick.  
- **Video Assembly:** If it looks good, we ship it. If not, we call it “experimental.”

---

> [!Note]
> The repo and this method is still under development and the results are not as good as I would have expected, so suggestions, issues and PRs are welcome. also suggest now methods for video interpolation!

---

## **In Case You’re Still Here**  
This repo is for those who like breaking things. No promises, no guarantees. Just cool experiments with diffusion models and video interpolation. If you get frustrated, that’s part of the process. Enjoy.

---
**“Fast code is better than clean code, and working code is the best kind.”**  
