<h2 align="center"> <a href="https://arxiv.org/abs/2411.19772">LongVALE: Vision-Audio-Language-Event Benchmark Towards Time-Aware Omni-Modal Perception of Long Videos</a></h2>

<h3 align="center"> Tiantian Geng, Jinrui Zhang, Qingni Wang, Teng Wang, Jinming Duan, Feng Zheng </h3>

<h5 align="center"> If our project helps you, please give us a star ⭐ and cite our <a href="#Citation">paper</a>!</h2>
<!-- # LongVALE -->

<!-- [![arxiv](https://img.shields.io/badge/Arxiv-2410.05643-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.19772) -->

[[🌐 Project Page]()] [[📖 arXiv Paper](https://arxiv.org/abs/2411.19772)] [[📊 LongVALE Dataset]()]

## News

<!-- - 28/02/2025, 🔥The LongVALE dataset is released. -->
- 27/02/2025, 🔥LongVALE has been accepted to CVPR 2025.

TODO

- [x] Release the annotation files of LongVALE.
- [ ] Release the extracted features (video, audio, speech) of LongVALE.
- [ ] Release the LongVALE-LLM model with training and evauluation code.
- [ ] Release raw videos of LongVALE.
  
## 👀 Overview
Recent advancements in video understanding remain limited to coarse-grained and visual-only tasks. However, real-world videos encompass omnimodal information (vision, audio, and speech) with a series of events forming a cohesive storyline. The lack of
multi-modal video data with fine-grained event annotations
and the high cost of manual labeling are major obstacles
to comprehensive omni-modality video perception. To address this gap, 
- we propose an automatic pipeline consisting of high-quality multi-modal video filtering, semantically coherent omni-modal event boundary detection, and crossmodal correlation-aware event captioning. 
- we present LongVALE, the first-ever Vision-Audio-Language
Event understanding benchmark comprising 105K omnimodal events with precise temporal boundaries and detailed relation-aware captions within 8.4K high-quality long videos. 
- we build LongVALE-LLM to enable video large language models (LLMs) for omni-modality fine-grained temporal video understanding for the first time. 
<div align="center">
    <img src="fig1_new3.png" alt="Overview of TRACE" width="700"/>
    <br/>
    <figcaption>Overview of TRACE.</figcaption>
</div>

## Dataset 
### Annotation files of training and evaluation sets
| Split           | Download | # Videos | # Omni-modal Events | Video Duration |
|-----------------|----------|-----------------|-----------|----------------|
|Training set | [🤗 link](https://huggingface.co/datasets/ttgeng233/LongVALE/resolve/main/train_set_info_7240.json)| 7,240 | 91,863 | 473.8 hrs |
|Evaluation set | [🤗 link](https://huggingface.co/datasets/ttgeng233/LongVALE/resolve/main/test_set_info_1171.json)| 1,171 |13,867 | 75.6 hrs |


**[Note]** The json files include the information of video id (YouTube id), video duration, timestamps and detailed captions of each omni-modal events. You can download the raw videos on YouTube using the provided video ids.

### LongVALE-based dialogue data for LongVALE-LLM training 
| Tuning Stage          | Data Source | Download | # Videos | QA Dialogues | 
|-----------------|----------|-----------------|-----------|---|
|Omni boundary perception |LongVALE | [🤗 link](https://huggingface.co/datasets/ttgeng233/LongVALE/resolve/main/stage_2_train_7240.json) | 7240 | 7240 |
|          |LongVALE + [VTimeLLLM_stage2](https://github.com/huangb23/VTimeLLM)  | [🤗 link](https://huggingface.co/datasets/ttgeng233/LongVALE/resolve/main/stage_2_ours_add_vtimellm.json) | ~141K | ~154K
|Omni instruction tuning | LongVALE | [🤗 link](https://huggingface.co/datasets/ttgeng233/LongVALE/resolve/main/stage_3_train_only_ours.json) | 7240 | ~25.4K |
| | LongVALE + [VTimeLLLM_stage3](https://github.com/huangb23/VTimeLLM) |[🤗 link](https://huggingface.co/datasets/ttgeng233/LongVALE/resolve/main/stage_3_ours_add_vtimellm.json)| - |~61.4K|


## Acknowledgement
We are grateful for the following awesome projects: [VTimeLLM](https://github.com/huangb23/VTimeLLM)
  

## Citation
If you find our project are useful for your research, please consider citing:
```
@article{geng2024longvale,
  title={Longvale: Vision-audio-language-event benchmark towards time-aware omni-modal perception of long videos},
  author={Geng, Tiantian and Zhang, Jinrui and Wang, Qingni and Wang, Teng and Duan, Jinming and Zheng, Feng},
  journal={arXiv preprint arXiv:2411.19772},
  year={2024}
}
```
