@echo off
set GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/stable-diffusion-v1-5/

del models\StableDiffusion\stable-diffusion-v1-5\*.ckpt
del models\StableDiffusion\stable-diffusion-v1-5\*.safetensors
del models\StableDiffusion\stable-diffusion-v1-5\safety_checker\*.safetensors
del models\StableDiffusion\stable-diffusion-v1-5\safety_checker\*.bin
del models\StableDiffusion\stable-diffusion-v1-5\text_encoder\*.safetensors
del models\StableDiffusion\stable-diffusion-v1-5\text_encoder\*.bin
del models\StableDiffusion\stable-diffusion-v1-5\unet\*.safetensors
del models\StableDiffusion\stable-diffusion-v1-5\unet\*.bin
del models\StableDiffusion\stable-diffusion-v1-5\vae\*.safetensors
del models\StableDiffusion\stable-diffusion-v1-5\vae\*.bin

rmdir /s /q models\StableDiffusion\stable-diffusion-v1-5\.git\

curl -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/pytorch_model.bin --output models\StableDiffusion\stable-diffusion-v1-5\text_encoder\pytorch_model.bin
curl -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin --output models\StableDiffusion\stable-diffusion-v1-5\unet\diffusion_pytorch_model.bin
curl -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.bin --output models\StableDiffusion\stable-diffusion-v1-5\vae\diffusion_pytorch_model.bin

move models\StableDiffusion\stable-diffusion-v1-5\* models\StableDiffusion

move models\StableDiffusion\stable-diffusion-v1-5\feature_extractor models\StableDiffusion
move models\StableDiffusion\stable-diffusion-v1-5\safety_checker models\StableDiffusion
move models\StableDiffusion\stable-diffusion-v1-5\scheduler models\StableDiffusion
move models\StableDiffusion\stable-diffusion-v1-5\text_encoder models\StableDiffusion
move models\StableDiffusion\stable-diffusion-v1-5\tokenizer models\StableDiffusion
move models\StableDiffusion\stable-diffusion-v1-5\unet models\StableDiffusion
move models\StableDiffusion\stable-diffusion-v1-5\vae models\StableDiffusion

rmdir /s /q models\StableDiffusion\stable-diffusion-v1-5\

curl -L https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt --output models\Motion_Module\v3_sd15_adapter.ckpt
curl -L https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt --output models\Motion_Module\v3_sd15_mm.ckpt