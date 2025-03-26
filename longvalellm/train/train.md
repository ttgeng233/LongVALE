# Training LongVALE-LLM

LongVALE-LLM adopts omni-modal boundary perception and instruction tuning to allow omni-modal event understanding. 

## Omni-modal boundary perception tuning
* Download [Vicuna v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) and [vtimellm_stage1](https://huggingface.co/datasets/ttgeng233/LongVALE/blob/main/checkpoints/vtimellm_stage1_mm_projector.bin) weights, and place them into the `checkpoints` directory.
* Download the dataset from [longvale-sft-bp-154k.json](https://huggingface.co/datasets/ttgeng233/LongVALE/blob/main/longvale-sft-bp-154k.json), and place it into the `data` directory.
* Download the pre-extracted features of LongVALE from [features_training](https://huggingface.co/datasets/ttgeng233/LongVALE/tree/main/features_training), and unzip them. Note that for visual features, you also need to download the [stage2 features](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2Ffeat&mode=list) of [VTimeLLM](https://github.com/huangb23/VTimeLLM) and add them to the folder of visual features. 
* Perform omni-modal boundary perception tuning, and make sure to `--feat_folder`, `--audio_feat_folder`, `--asr_feat_folder` in the script to the corresponding feature folder for each modality.
```shell
cd LongVALE
bash scripts/boundary_perception.sh
```

## Omni-modal instruction tuning
* Download the dataset from [longvale-sft-it-61k.json](https://huggingface.co/datasets/ttgeng233/LongVALE/blob/main/longvale-sft-it-61k.json), and place it into the `data` directory.
* Maintain the audio and speech feature folders, and make a new visual feature folder for this training stage. Download the [stage3 features](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2Ffeat&mode=list) of [VTimeLLM](https://github.com/huangb23/VTimeLLM), and palce them and visual features of LongVALE to the new folder together.
* Perform omni-modal instruction tuning, and make sure to `--feat_folder`, `--audio_feat_folder`, `--asr_feat_folder` in the script to the corresponding feature folder for each modality.
```shell
bash scripts/instruction_tuning.sh
```
The folder structure should look like:
```markdown
- LongVALE
    - checkpoints
        - vicuna-7b-v1.5
        - vtimellm_stage1_mm_projector.bin
    - data 
        - longvale-sft-bp-154k.json
        - longvale-sft-it-61k.json
    - scripts
        - boundary_perception.sh
        - instruction_tuning.sh
        - ...
    - longvalellm
    - ...
```