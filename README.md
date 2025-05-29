# Blood Cell Classification with Deep Learning

**Course**: Artificial Neural Networks and Deep Learning (AN2DL)  
**Team Members**: Denis Sanduleanu, Jonatan Sciaky, Alessio Caggiano, Mattia Repetti
**Project Group Name**: *Gan’s n Roses*  

## 📌 Overview

This project focuses on classifying images of blood cells into 8 categories using state-of-the-art Deep Learning techniques. Starting from a custom CNN and evolving through transfer learning and advanced augmentation strategies, we progressively improved our model's generalization and accuracy.

## 🧠 Problem

- **Dataset**: 13,759 RGB images (96×96) across 8 blood cell classes.
- **Challenges**:
  - Class imbalance.
  - Limited dataset size.
  - Generalization gap between local results and external test platform.

## 🛠️ Approach

We followed an incremental approach:
1. **Custom CNN** – Initial attempt, later abandoned due to poor generalization.
2. **Transfer Learning** – Used pre-trained ConvNeXt models (Small, Base, Large).
3. **Augmentation** – Implemented advanced techniques like:
   - CutMix
   - MixUp
   - RandAugment
   - GridMask
   - CutOut
   - Custom masking
4. **Regularization & Optimization**:
   - Dropout layers
   - L1/L2 regularization
   - Early stopping
   - Optimizers: Adam, Nadam, AdamW, Lion

## 📊 Results

Best performance achieved with `ConvNeXtLarge`, trained with the following setup:
- Batch size: 256
- Training set: 42,000 images (augmented)
- Validation set: Augmented with RandAug (70%)
- Fine-tuning: Last 25–40% of layers trainable
- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-5`)
- Dropout: 35%
- Regularization: L1L2 (`1e-3`)
- EarlyStopping: Patience = 4 (fine-tuning phase)

**Final accuracy on the platform**: **83%**

## 📦 Repository Contents

- `PreProcess.ipynb`: Notebook for dataset cleaning, outlier removal, and preprocessing (e.g., masking).
- `balanced_augmentation.ipynb`: Implements and tests various advanced data augmentation strategies including RandAug, CutMix, and MixUp to handle class imbalance and improve generalization.
- `custom.ipynb`: A custom CNN architecture built from scratch, used as a baseline for comparison.
- `convnextbase.ipynb`: Model training and evaluation using the ConvNeXt Base architecture with transfer learning.
- `ConvNeXtSmall.ipynb`: Implementation and tuning of ConvNeXt Small for classification.
- `convNextlarge.ipynb`: Final and best-performing model using ConvNeXt Large with extensive augmentation, fine-tuning, and regularization.
- `Report1.pdf`: Full project report with methodology, results, analysis, and conclusions.


## 📚 References

- [CutMix, MixUp, and RandAugment in KerasCV](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/)
- [Explainable AI for blood cell classification](https://www.sciencedirect.com/science/article/pii/S2153353924000282)

## 🔍 Future Work

- Implement model ensembling to further boost accuracy.
- Extend classifier to other types of cell classification tasks.
- Integrate explainability techniques for medical usability.
