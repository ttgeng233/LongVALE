from PIL import Image
from io import BytesIO
import base64
import numpy as np
import torch
import decord
from transformers import StoppingCriteria
from longvalellm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moviepy.editor import VideoFileClip
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi
from transformers import WhisperFeatureExtractor

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors='pt')['pixel_values']


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    elif tokenizer.name == "GLMTokenizer":
        offset = 2
        input_ids = prompt_chunks[0][:2]

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:].equal(keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        # print(_, param.requires_grad, param.numel())
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

class VideoExtractor():
    """Dataset for supervised fine-tuning."""

    def __init__(self, N=100):
        self.N = N

    def extract(self, data):
        video_path = data['video']
        id = data['id']
        
        try:
            video_reader = decord.VideoReader(video_path)
            total_frames = len(video_reader)
            start = 0
            end = total_frames - 1

            split = data.get('split', None)
            if split is not None:
                fps = video_reader.get_avg_fps()
                start = max(int(fps * split[0]), 0)
                end = min(int(fps * split[1]), total_frames - 1)
            sampled_indices = np.linspace(start, end, self.N, dtype=np.int32)
            sampled_frames = video_reader.get_batch(sampled_indices).asnumpy()
        except Exception as e:
            print(e)
            return None, torch.zeros(1)
        
        images = torch.from_numpy(sampled_frames.transpose((0, 3, 1, 2)))
        return id, images
    

class BEATSAudioExtractor():
    """
    Adapted from https://github.com/Yui010206/CREMA/blob/main/lavis/processors/audio_processors.py
    """

    def __init__(self, model_name='', sampling_rate=16000, n_frames=2, frame_length=512, is_eval=False):
        self.model_name = model_name
        self.sampling_rate = sampling_rate
        self.n_frames = n_frames
        self.frame_length = frame_length
        self.fbank_mean = 15.41663
        self.fbank_std = 6.55582
        self.is_eval = is_eval


    def extract(self, data):

        def empty_audio_tensor():
            return torch.zeros((self.n_frames, self.frame_length, 128))
        aupath = data['video']
        split = data.get('split', None)
        start_sec = None
        end_sec = None
        if split is not None:
            start_sec = split[0]
            end_sec = split[1]

            # Handle MP4 files
        if aupath.endswith('.mp4'):
            video = VideoFileClip(aupath)
            if start_sec is not None and end_sec is not None:
                video = video.subclip(start_sec, end_sec)
            try:
                audio_np = video.audio.to_soundarray(fps=self.sampling_rate)
            except:
                print(aupath, 'can not extract')
            # import librosa, os
            # root_path = '' #for audios that cannot be loaded.
            # wav_path = os.path.join(root_path, os.path.basename(aupath).split('.')[0] + '.wav')
            # audio_np, sampling_rate = librosa.load(wav_path, sr=self.sampling_rate)
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)  # Convert to mono
            waveform = torch.tensor(audio_np).float()
            sr = self.sampling_rate
        else:
            waveform, sr = torchaudio.load(aupath)

        # print('waveform', waveform.shape) # [channel, time] 
        # Validate waveform
        if len(waveform.shape) == 0:
            return empty_audio_tensor()

        # Convert stereo to mono
        if waveform.shape[0] == 2:
            waveform = torch.mean(waveform, dim=0)

        # Resample waveform if necessary
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            waveform = resampler(waveform)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform * 2**15

        # Compute fbank features
        try:
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=self.sampling_rate,
                frame_length=25,
                frame_shift=10, # 10
            )
            fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)
        except:
            return empty_audio_tensor()

        # Handle padding and frames extraction differently for eval and training modes
        if not self.is_eval:
            fbank_pad_len = self.frame_length * self.n_frames - fbank.shape[0]
            if fbank_pad_len > 0:
                fbank = torch.nn.ZeroPad2d((0, 0, 0, fbank_pad_len))(fbank)
            fbank = fbank[:self.frame_length * self.n_frames]
            frames = [fbank[i*self.frame_length:(i+1)*self.frame_length].unsqueeze(0) for i in range(self.n_frames)]
        else:
            fbank_pad_len = fbank.shape[0] % self.frame_length
            if fbank_pad_len > 0:
                fbank = torch.nn.ZeroPad2d((0, 0, 0, fbank_pad_len))(fbank)
            curr_frames = fbank.shape[0] // self.frame_length
            frames = [fbank[i*self.frame_length:(i+1)*self.frame_length].unsqueeze(0) for i in range(curr_frames)]

        return torch.cat(frames, dim=0)

class SpeechExtractor():
    def __init__(self, model, sampling_rate=16000):
        self.sample_rate = sampling_rate
        self.transform = WhisperFeatureExtractor.from_pretrained(model)

    def extract(self, data):
        aupath = data['video']
        split = data.get('split', None)
        start_sec = None
        end_sec = None
        if split is not None:
            start_sec = split[0]
            end_sec = split[1]
        if aupath.endswith('.mp4'):
            video = VideoFileClip(aupath)
            if start_sec is not None and end_sec is not None:
                video = video.subclip(start_sec, end_sec)
            try:
                audio_np = video.audio.to_soundarray(fps=self.sample_rate)
            except:
                print(aupath, 'cannot extract')
            # import librosa, os
            # root_path = '' # for audios that cannot be loaded.
            # wav_path = os.path.join(root_path, os.path.basename(aupath).split('.')[0] + '.wav')
            # audio_np, sampling_rate = librosa.load(wav_path, sr=self.sample_rate)

            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)  # Convert to mono
            waveform = torch.tensor(audio_np).float()
            sr = self.sample_rate
        else:
            waveform, sr = torchaudio.load(aupath)

        if len(waveform) > 30 * self.sample_rate:
            audio_list = [waveform[i: i + 30 * self.sample_rate] for i in range(0, len(waveform), 30 * self.sample_rate)]
            spectrogram_list = []
            for audio_piece in audio_list:
                spectrogram_piece = self.transform(
                    audio_piece,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    max_length=30 * self.sample_rate,
                )
                spectrogram_list.append(spectrogram_piece["input_features"].squeeze())
            spectrogram = torch.stack(spectrogram_list, dim=0)
        else:
            spectrogram = self.transform(
                waveform,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                max_length=30 * self.sample_rate,
                )
            spectrogram = spectrogram["input_features"].squeeze()

        return spectrogram