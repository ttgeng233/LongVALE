import sys
sys.path.append('./')
from longvalellm.mm_utils import SpeechExtractor
from transformers.models.whisper.modeling_whisper import WhisperModel
from torch.utils.data import Dataset, DataLoader 
import torch
import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def prepare_model(whisper_path, gpu_id):
    device = torch.device('cuda:{}'.format(gpu_id))
    speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
    speech_encoder = speech_encoder.to(device)
    return speech_encoder, device    


class AudioDataset(Dataset):
    def __init__(self, annotation, video_dir, model):
        with open(annotation, 'r') as f:
            self.data = json.load(f)
        self.processor = SpeechExtractor(model)
        self.video_dir = video_dir

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        video_id = item['id']
        video_path = os.path.join(self.video_dir, '{}.mp4'.format(video_id))
        sample = {'video': video_path}
        spectrogram = self.processor.extract(sample)
        return spectrogram, video_id


def collate_fn(batch):
    spectrogram, video_ids = zip(*batch)
    spectrogram = torch.cat(spectrogram, dim=0)
    video_ids = list(video_ids)
    return spectrogram, video_ids


# DataLoader
def create_data_loader(annotation, video_dir, model, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = AudioDataset(annotation, video_dir, model)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str, default='./data/longvale-annotations-eval.json')
    parser.add_argument("--video_dir", type=str, default='./raw_videos')
    parser.add_argument("--output_dir", type=str, default='./output/speech_features')
    parser.add_argument("--checkpoint", type=str, default='./checkpoints/openai/whisper-large-v2')
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model, device = prepare_model(args.checkpoint, args.gpu_id)
    data_loader = create_data_loader(args.annotation, args.video_dir, args.checkpoint)
    second_per_window = 5.12
    second_stride = 5.12
    with torch.no_grad():
        for (spectrogram, video_ids) in tqdm(data_loader):
            spectrogram = spectrogram.to(device)
            video_id = video_ids[0]
            save_path = os.path.join(args.output_dir, '{}.npy'.format(video_id))
            if os.path.exists(save_path):
                continue
                # bsz * n_frames, num_patches, dim
            if spectrogram.dim() == 2:  
                spectrogram = spectrogram.unsqueeze(0)  
            try:
                features = model(spectrogram, return_dict=True).last_hidden_state
            except:
                print(video_id, "can't extract speech feature.")
                continue
            # bsz * n_frames, dim
            dim = features.shape[-1]
            features = features.reshape(1, -1, dim)

            B, T, C = features.shape
            kernel = round(1500 * second_per_window / 30.0)
            stride = round(1500 * second_stride / 30.0)
            kernel = (1, kernel)
            stride = (1, stride)
            speech_embeds_tr = features.transpose(1, 2).unsqueeze(2)
            speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, stride=stride)
            _, _, L = speech_embeds_overlap.shape
            speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L) 
            speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1]) 
            speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C) 
            
            speech_embeds_pooled = torch.mean(speech_embeds, dim=1) 
            # FIXME, too long to train, consider pooling
            output_features = speech_embeds_pooled.cpu().numpy()
            np.save(save_path, output_features)

