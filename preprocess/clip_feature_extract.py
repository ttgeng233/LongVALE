import sys
sys.path.append('./')
import clip
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from longvalellm.mm_utils import VideoExtractor 
import json
import os
from tqdm import tqdm
import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC



def prepare_model(checkpoint, gpu_id):
    device = torch.device('cuda:{}'.format(gpu_id))
    model, _ = clip.load(checkpoint)
    return model, device


class AudioDataset(Dataset):
    def __init__(self, annotation, video_dir):
        with open(annotation, 'r') as f:
            self.data = json.load(f)
        self.processor = VideoExtractor(N=100)
        self.video_dir = video_dir
        self.transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258,     0.27577711)),
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        #activitynet 
        item = self.data[index]
        video_id = item['id']
        video_path = os.path.join(self.video_dir, '{}.mp4'.format(video_id))
        sample = {'id' : video_id, 'video': video_path}
        _, images = self.processor.extract(sample)
        images = self.transform(images / 255.0)
        return images, video_id
    

def collate_fn(batch):
    images, video_ids = zip(*batch)
    images = torch.cat(images, dim=0)
    video_ids = list(video_ids)
    return images, video_ids


# DataLoader
def create_data_loader(annotation, video_dir, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = AudioDataset(annotation, video_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str, default='./data/longvale-annotations-eval.json')
    parser.add_argument("--video_dir", type=str, default='./raw_videos')
    parser.add_argument("--output_dir", type=str, default='./output/video_features')
    parser.add_argument("--checkpoint", type=str, default='./checkpoints/ViT-L-14.pt')
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model, device = prepare_model(args.checkpoint, args.gpu_id)
    data_loader = create_data_loader(args.annotation, args.video_dir)
    with torch.no_grad():
        for (images, video_ids) in tqdm(data_loader):
            images = images.to(device)
            video_id = video_ids[0]
            save_path = os.path.join(args.output_dir, '{}.npy'.format(video_id))
            if os.path.exists(save_path):
                continue
            features = model.encode_image(images)
            features = features.cpu().numpy()
            np.save(save_path, features)