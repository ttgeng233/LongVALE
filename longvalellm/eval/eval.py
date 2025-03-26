import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
import sys
sys.path.append(root_dir)

import re
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from longvalellm.model.builder import load_pretrained_model
from longvalellm.utils import disable_torch_init
from longvalellm.inference import inference

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm_stage1_mm_projector.bin")
    parser.add_argument("--pretrain_audio_mlp_adapter", type=str, default=None)
    parser.add_argument("--pretrain_asr_mlp_adapter", type=str, default=None) 
    parser.add_argument("--stage2", type=str, default="checkpoints/longvalellm-vicuna-v1-5-7b/longvale-vicuna-v1-5-7b-stage2-bp")
    parser.add_argument("--stage3", type=str, default="checkpoints/longvalellm-vicuna-v1-5-7b/longvale-vicuna-v1-5-7b-stage3-it")
    parser.add_argument("--model_base", type=str, default="checkpoints/vicuna-7b-v1.5")
    parser.add_argument("--data_path", type=str, default="/path/to/longvale-annotations-eval.json")
    parser.add_argument("--video_feat_folder", type=str, default="/path/to/features_eval/visual_features_1171")
    parser.add_argument("--audio_feat_folder", type=str, default="/path/to/features_eval/audio_features_1171")
    parser.add_argument("--asr_feat_folder", type=str, default="/path/to/features_eval/speech_features_1171") 
    parser.add_argument("--task", type=str, default='all', choices=['all', 'grounding', 'captioning', 'seg_captioning'])
    parser.add_argument("--log_path", type=str, default='longvalellm/eval/log/example_log.txt')
    args = parser.parse_args()
    return args

def iou(outputs, gt):
    matches = re.search(r"(\d{2}) (to|and) (\d{2})", outputs)
    if not matches:
        return 0
    from_number = float(matches.group(1)) / 100
    to_number = float(matches.group(3)) / 100
    s, e = gt
    intersection = max(0, min(to_number, e) - max(from_number, s))
    union = max(to_number, e) - min(from_number, s)
    iou = intersection / union
    return round(iou, 2)


def write_log(log_path, video_id, task, query_id, answer, info=None):
    log = {
        'video_id': video_id,
        'task': task,
        'query_id': query_id,
        'answer': answer
    }
    if info is not None:
        log['info'] = info
    with open(log_path, 'a') as f:
        f.write(json.dumps(log) + '\n')

questions = {
    'grounding': ['At which time interval can we find {} taking place in the video? Give the timestamps in the fromat: From xx to xx.'], 
    'captioning': ['Could you please detail the events that took place during different time segments in the video? List the events in the format: From xx to xx, event1. \n From xx to xx, event2. \n ...'],
    'seg_captioning': ['Can you describe what occurred from <start> to <end> in the video? Please give the event description directly.']
} 

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    model.to(torch.float16)

    js = json.load(open(args.data_path))
    for id, data in tqdm(js.items()):
        video_features = None
        audio_features = None
        asr_features = None

        if args.video_feat_folder is not None:
            video_feat_path = os.path.join(args.video_feat_folder, f"{id}.npy")
            if os.path.isfile(video_feat_path):
                video_features = torch.from_numpy(np.load(video_feat_path)).cuda()
        
        if args.audio_feat_folder is not None:
            audio_feat_path = os.path.join(args.audio_feat_folder, f"{id}.npy")
            if os.path.isfile(audio_feat_path):
                audio_features = torch.from_numpy(np.load(audio_feat_path)).cuda()
        
        if args.asr_feat_folder is not None:
            asr_feat_path = os.path.join(args.asr_feat_folder, f"{id}.npy")
            if os.path.isfile(asr_feat_path):
                asr_features = torch.from_numpy(np.load(asr_feat_path)).cuda()

        if video_features is None:
            print(f'Can not find video {id}')
            continue
 
        if args.task in ['captioning', 'all']:
            for query_id, query in enumerate(questions['captioning']):
                answer = inference(model, video_features, audio_features, asr_features, "<video>\n " + query, tokenizer)
                write_log(args.log_path, id, 'captioning', query_id, answer)
      
        if args.task in ['grounding', 'all']:
            for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
                sentence = sentence.strip().lower()
                if sentence.endswith("."):
                    sentence = sentence[:-1]

                for query_id, query in enumerate(questions['grounding']):
                    answer = inference(model, video_features, audio_features, asr_features, "<video>\n" + query.format(sentence), tokenizer)
                    gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
                    u = iou(answer, gt)
                    write_log(args.log_path, id, 'grounding', query_id, answer, info={"sentence_id": sentence_id, 'iou': u})
        
        if args.task in ['seg_captioning', 'all']:
            def convert(duration, x):
                x = x / duration * 100
                x = str(min(round(x), 99))
                if len(x) == 1:
                    x = "0" + x
                return x
            for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
                start_time = convert(data['duration'], timestamps[0])
                end_time = convert(data['duration'], timestamps[1])

                for query_id, query in enumerate(questions['seg_captioning']):
                    query = query.replace('<start>', start_time)
                    query = query.replace('<end>', end_time)
                    answer = inference(model, video_features, audio_features, asr_features, "<video>\n " + query, tokenizer)
                    write_log(args.log_path, id, 'seg_captioning', query_id, answer, info={"sentence_id": sentence_id})