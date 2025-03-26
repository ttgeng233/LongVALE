# LongVALE-LLM Evaluation

TODO
- [x] Release evaluation code on LongVALE benchmark.
- [ ] Release evaluation code on AVSD and Music-AVQA benchmarks.

We provide evaluation code for LongVALE-LLM on omni-modal temporal video grouding, dense video captioning and segment captioning tasks on our proposed LongVALE benchmark. We outline the evaluation process as follows.

### Download model weights
- Download [Vicuna v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) and [vtimellm_stage1](https://huggingface.co/datasets/ttgeng233/LongVALE/blob/main/checkpoints/vtimellm_stage1_mm_projector.bin) weights, and place them into the `checkpoints` directory.
- Download LongVALE-LLM model from [longvalellm-vicuna-v1-5-7b.tar.gz](https://huggingface.co/datasets/ttgeng233/LongVALE/blob/main/checkpoints/longvalellm-vicuna-v1-5-7b.tar.gz), and place it into the `checkpoints` directory.
  
### Download data
- Download the annotation JSON file for LongVALE test set from [longvale-annotations-eval.json](https://huggingface.co/datasets/ttgeng233/LongVALE/blob/main/longvale-annotations-eval.json).
- Download pre-extracted features of LongVALE test set from [features_eval](https://huggingface.co/datasets/ttgeng233/LongVALE/tree/main/features_eval), and unzip them.

### Run the evaluation code
- Run `longvalellm/eval/eval.py`, and record the model's response in a log file.
```bash
python longvalellm/eval/eval.py --data_path /path/to/longvale-annotations-eval.json --video_feat_folder /path/to/features_eval/visual_features_1171 --audio_feat_folder /path/to/features_eval/audio_features_1171 --asr_feat_folder /path/to/features_eval/speech_features_1171 --task all --log_path /path/to/log
```
- In order to compute metrics for the dense video captioning task, you need to install `pycocoevalcap` and `Java`. 

```bash
pip install pycocoevalcap
conda install conda-forge::openjdk
```

- Parse the log file and obtain the corresponding metrics.

```bash
python longvalellm/eval/metric.py --data_path /path/to/longvale-annotations-eval.json --log_path /path/to/log
```