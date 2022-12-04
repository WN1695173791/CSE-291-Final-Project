# UCSD FA22 CSE291 Deep Generative Model Course Project

The code is based on [Guided-Diffusion](https://github.com/openai/guided-diffusion)
It requires PyTorch 1.11 and Faiss 1.7.3.

To reproduce the results:
1. Download LSUN-Cat, LSUN-Horse, Celeb-A, FFHQ, and Pascal VOC 2012 to ``datasets/``
2. Train contrastive finetuning model with ``python contrastive.py``. Edit the path to dataset in the file accordingly.
3. Online clustering and evaluate with ``python eval.py``. By default, the file evaluate on Celeb-A with 8 keypoints. Edit the setting in the file accordingly.
4. Figure in the report are drawn with ``figures.ipynb``
