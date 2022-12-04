# UCSD FA22 CSE291 Deep Generative Model Course Project

The code is based on [Guided-Diffusion](https://github.com/openai/guided-diffusion)
It requires PyTorch 1.11 and Faiss 1.7.3.

To reproduce the results:
1. Download LSUN-Cat, LSUN-Horse, Celeb-A, FFHQ, and Pascal VOC 2012 to ``datasets/``
2. Download the pretrained DDPM checkpoints to ``checkpoints`` from [Guided-Diffusion](https://github.com/openai/guided-diffusion).
3. Train contrastive finetuning model with ``python contrastive.py``. Edit the path to the dataset in the file accordingly.
4. Online clustering and evaluate with ``python eval.py``. By default, the file evaluate on Celeb-A with 8 keypoints. Edit the settings in the file accordingly.
5. Figure in the report are drawn with ``figures.ipynb``.
