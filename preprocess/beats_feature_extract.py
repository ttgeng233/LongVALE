import sys
sys.path.append('./')
from longvalellm.mm_utils import BEATSAudioExtractor
from longvalellm.model.beats.BEATs import BEATs, BEATsConfig
import torch
import argparse
from torch.utils.data import Dataset, DataLoader 
import json
import os
from tqdm import tqdm
import numpy as np



def prepare_model(checkpoint_path, gpu_id):
    # checkpoint_path = '.cache/BEATs_iter3_plus_AS20K.pt'
    device = torch.device('cuda:{}'.format(gpu_id))
    checkpoint = torch.load(checkpoint_path)
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, device


class AudioDataset(Dataset):
    def __init__(self, annotation, video_dir):
        with open(annotation, 'r') as f:
            self.data = json.load(f)
        self.processor = BEATSAudioExtractor(is_eval=True)
        self.video_dir = video_dir

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        video_id = item['id']
        video_path = os.path.join(self.video_dir, '{}.mp4'.format(video_id))
        sample = {'video': video_path}
        fbank = self.processor.extract(sample)
        return fbank, video_id
    

def collate_fn(batch):
    fbank, video_ids = zip(*batch)
    fbank = torch.cat(fbank, dim=0)
    video_ids = list(video_ids)
    return fbank, video_ids


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
    parser.add_argument("--output_dir", type=str, default='./output/audio_features')
    parser.add_argument("--checkpoint", type=str, default='./checkpoints/BEATs_iter3_plus_AS20K.pt')
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model, device = prepare_model(args.checkpoint, args.gpu_id)
    data_loader = create_data_loader(args.annotation, args.video_dir)
    with torch.no_grad():
        for (fbank, video_ids) in tqdm(data_loader):
            fbank = fbank.to(device)
            video_id = video_ids[0]
            save_path = os.path.join(args.output_dir, '{}.npy'.format(video_id))
            if os.path.exists(save_path):
                continue
            # bsz * n_frames, num_patches, dim
            features = model.extract_features(fbank)[0]
            # bsz * n_frames, dim
            features = features.mean(dim=1).squeeze(1)
            features = features.cpu().numpy()
            np.save(save_path, features)