
# ABVnet: Adapter-based fine-tuning Bimodal ViTs network for Dynamic Facial Expression Recognition in-the-wild

This repository provides an official implementation for the paper ABVnet: Adapter-based fine-tuning Bimodal ViTs network for Dynamic Facial Expression Recognition in-the-wild



## Installation

Please create an environment with Python 3.10 and use requirements file to install the rest of the libraries

```bash
pip install -r reqiurements.txt
```

## Data preparation

We provide the codes for [DFEW](https://dfew-dataset.github.io/) and [MAFW](https://mafw-database.github.io/MAFW/) datasets, which you would need to download. Then, please refer to DFER-CLIP repository for transforming the annotations that are provided in annotations/ folder to your own paths. To extract faces from MAFW dataset, please refer to data_utils that has an example of face detection pipeline. 

You will also need to download pre-trained checkpoints for vision encoder from [https://github.com/FuxiVirtualHuman/MAE-Face](https://github.com/FuxiVirtualHuman/MAE-Face/releases) and for audio encoder from [https://github.com/facebookresearch/AudioMAE](https://drive.usercontent.google.com/download?id=1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu&export=download&authuser=0) Please extract them and rename the audio checkpoint to 'audiomae_pretrained.pth'. Both checkpoints are expected to be in root folder.

## Running the code

The main script in main.py. You can invoke it through running:
```bash
./train_DFEW.sh
```
```bash
./train_MAFW.sh
```



## References
This repository is based on MMA-DFER https://github.com/katerynaCh/MMA-DFER. We also thank the authors of MAE-Face https://github.com/FuxiVirtualHuman/MAE-Face and Audiomae https://github.com/facebookresearch/AudioMAE


