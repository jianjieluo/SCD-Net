# Semantic-Conditional Diffusion Networks for Image Captioning

## Introduction
This is the official repository for **Semantic-Conditional Diffusion Networks for Image Captioning**. 

SCD-Net is a cascaded diffusion captioning model with a novel semantic-conditional diffusion process that upgrades conventional diffusion model with additional semantic prior. A novel guided self-critical sequence training strategy is further devised to stabilize and boost the diffusion process. To our best knowledge, SCD-Net is the first diffusion-based captioning model that achieves better performance than the naive auto-regressive transformer captioning model **conditioned on the same visual features(i.e. [bottom-up attention region features](https://github.com/peteanderson80/bottom-up-attention)) in both XE and RL training stages.** SCD-Net is also **the first diffusion-based captioning model that adopts CIDEr-D optimization successfully** via a novel guided self-critical sequence training strategy. SCD-Net achieves state-of-the-art performance among non-autoregressive/diffusion captioning models and comparable performance aginst the state-of-the-art autoregressive captioning models, which indicates the promising potential of using diffusion models in the challenging image captioning task.

## Framework
![scdnet](imgs/scdnet.png)

## Data Preparation



## Training



## Inference




## Citation



## Acknowledgement



## License

MIT