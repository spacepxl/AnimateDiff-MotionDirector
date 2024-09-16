import os
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image


def calculate_size(h, w, base_res=512, step=64):
    total_res = base_res ** 2
    aspect_ratio = w / h
    
    new_height = round(((total_res / aspect_ratio) ** 0.5) / step) * step
    new_width  = round(((total_res * aspect_ratio) ** 0.5) / step) * step
    
    return new_height, new_width


def load_images_from_folder(folder, base_res=512, crop=False):
    path_list = glob(os.path.join(folder, "*.png"))
    if len(path_list) == 0:
        path_list = glob(os.path.join(folder, "*.jpg"))
    
    frames = []
    for path in tqdm(path_list, desc="loading frames", leave=False):
        image_tensor = read_image(path) # C H W
        frames.append(image_tensor)
    frames = torch.stack(frames, dim=0) # B C H W
    
    if crop:
        transforms = v2.Compose([
            v2.ToDtype(torch.float16, scale=True),
            v2.CenterCrop(min(frames.size(2), frames.size(3))),
            v2.Resize(base_res),
            ])
    else:
        h, w = calculate_size(frames.size(2), frames.size(3), base_res=base_res)
        transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((h, w)),
            ])
    
    return transforms(frames)


def encode_image_batch(images, vae):
    with torch.no_grad():
        latents = []
        for i in tqdm(range(images.size(0)), desc="encoding frames", leave=False):
            frame = images[i].unsqueeze(0).to(vae.device) * 2 - 1
            frame = vae.encode(frame).latent_dist.sample()
            latents.append(frame * 0.18215)
        return torch.cat(latents, dim=0).to("cpu")


def encode_text_prompt(prompt, tokenizer, text_encoder):
    with torch.no_grad():
        prompt_ids = tokenizer(
            prompt, 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(text_encoder.device)
        return text_encoder(prompt_ids)[0].to("cpu")


class VideoFrameDataset(Dataset):
    def __init__(
        self, 
        root_folder, 
        vae = None,
        tokenizer = None,
        text_encoder = None,
        default_caption: str = "",
        base_res: int = 512,
        crop: bool = False,
    ):
        self.root_folder = root_folder
        self.videos = []
        self.captions = []
        
        for folder in tqdm(os.listdir(self.root_folder), desc="loading videos"):
            folder = os.path.join(self.root_folder, folder)
            
            if os.path.isdir(folder):
                frames = load_images_from_folder(folder, base_res=base_res, crop=crop)
                if vae is not None:
                    frames = encode_image_batch(frames, vae)
                self.videos.append(frames)
                
                caption_files = glob(os.path.join(folder, "*.txt"))
                caption = default_caption
                if len(caption_files) > 0:
                    with open(caption_files[0], 'r') as t:
                        caption = t.read()
                if tokenizer is not None and text_encoder is not None:
                    caption = encode_text_prompt(caption, tokenizer, text_encoder)
                self.captions.append(caption)
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        return self.videos[idx], self.captions[idx]