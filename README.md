<div align="center">
  <img src="assets/readme_images/training_arch.png" alt="Background Image" width="95%" />
</div>

<h1 align="center"> Semi-Supervised Semantic Image Segmentation with Self-correcting Networks </h1>

<p align="center">
  A pytorch implementation of the <strong>CVPR 2020 paper</strong>, <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Ibrahim_Semi-Supervised_Semantic_Image_Segmentation_With_Self-Correcting_Networks_CVPR_2020_paper.pdf"><em>Semi-Supervised Semantic Image Segmentation with Self-correcting Networks</em></a> by <strong>Ibrahim et al.</strong>
  This framework enables joint learning from a small fully supervised dataset and large weakly labeled data, using self-correcting networks to refine predictions and reduce annotation noise 
</p>


## Table of Content
1. [Stage 1: Ancillary Model Training](#stage-1-ancillary-model-training)
    - [Model Architecture](#model-architecture)
    - [Loss Function](#loss-function)
    - [Hyperparameters](#hyperparameteres)
    - [Training Setup](#training-setup)
    - [Training Dataset](#training-dataset)
2. [Stage 2](#stage-2)


<strong>The approach in the paper divides training into three stages: the first trains the ancillary model, the second trains the self-correcting network, and the third focuses on the primary model.</strong>
