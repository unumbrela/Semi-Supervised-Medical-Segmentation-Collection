# **A Collection of Semi-Supervised Medical Image Segmentation Methods**

🇨🇳 [中文](README.md) | 🇺🇸 [English](README-EN.md)

## **📖 Research Direction Introduction**

**Semi-supervised medical image segmentation** is an important and active research field that primarily addresses the problem of scarce labeled data in medical image segmentation.

### **🎯 Research Background**

Deep neural networks offer state-of-the-art performance for automated medical image segmentation, but training these models requires expensive annotation work by experts. Due to the difficulty of obtaining large amounts of annotated data, semi-supervised learning is emerging as an attractive solution for medical image segmentation.

### **🔧 Core Technical Frameworks**

These methods primarily rely on the following core technologies:

- **Mean Teacher (MT) Architecture**: A teacher-student framework where the teacher model updates its parameters via exponential moving average (EMA) to guide the student model's learning.
- **Consistency Regularization**: Imposes data-level and model-level consistency on unlabeled data to enforce consistent predictions across different perturbations.
- **Uncertainty Guidance**: Incorporates uncertainty maps into the model to force the student to learn from the teacher's high-confidence predictions.

### **🏥 Application Scenarios**

These methods have been applied to various medical image segmentation tasks, such as cardiac magnetic resonance imaging, prostate segmentation, and brain tumor segmentation, demonstrating excellent performance on various medical datasets.

## **📚 Paper Method Overview**

Method Publication Year Conference/Journal Core Innovation GitHub Link

[Mean Teacher ](https://arxiv.org/abs/1703.01780)2018 NIPS Weighted average label prediction [GitHub](https://github.com/shunk031/chainer-MeanTeachers)

[UA-MT ](https://arxiv.org/abs/1907.07034)2019 MICCAI Uncertainty-guided Mean Teacher [GitHub](https://github.com/yulequan/UA-MT)

[DTC ](https://ojs.aaai.org/index.php/AAAI/article/view/17066)2021 AAAI Dual-task consistency learning [GitHub](https://github.com/HiLab-git/DTC)

[UPC ](https://www.sciencedirect.com/science/article/pii/S1746809422006577)2023 CBM Uncertainty-aware pseudo-labeling and consistency [GitHub](https://github.com/AIforMS/UPC-Pytorch)

[MCF ](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MCF_Mutual_Correction_Framework_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.pdf)2023 CVPR Mutual correction framework to address cognitive bias [GitHub](https://github.com/WYC-321/MCF)

[BCP ](https://arxiv.org/pdf/2305.00673)2023 CVPR Bidirectional copy-paste data augmentation [GitHub](https://github.com/DeepMed-Lab-ECNU/BCP)

[CML ](https://openreview.net/pdf/98a9ec17dea3e4d96eb45416cca53a8364aa93b1.pdf)2024 ACM MM Cross-Perspective Complementary Learning [GitHub](https://github.com/SongwuJob/CML)

[AC-MT ](https://www.sciencedirect.com/science/article/pii/S1361841523001408)2023 MedIA Fuzzy Consensus Mean Teacher [GitHub](https://github.com/lemoshu/AC-MT)

[DyCON ](https://arxiv.org/pdf/2504.04566)2025 CVPR Dynamic Uncertainty-Aware Consistency and Contrastive Learning [GitHub](https://github.com/KU-CVML/DyCON)

[V-Net ](https://arxiv.org/abs/1606.04797)2016 3DV 3D Medical Image Segmentation Foundation Network [GitHub](https://github.com/mattmacy/vnet.pytorch)

## **📋 Detailed Method Introduction**

### **Mean Teacher (MT)**

> **Paper link**: [Mean teachers are better role models](https://arxiv.org/abs/1703.01780)

> **GitHub repository link**: [MT](https://github.com/shunk031/chainer-MeanTeachers)

> **Year of publication**: 2018 NIPS

#### **Core innovation**

The Mean Teacher method replaces label prediction averages with model weight averages, solving the limitations of Temporal Ensembling:

- **Student model**: Normal training network, accepting gradient updates
- **Teacher model**: Exponential moving average (EMA) of student model weights
- 
- Both models apply different noise perturbations to the input

**Technical details**

- **EMA update**:`θ'_t = αθ'_{t-1} + (1-α)θ_t`
- **Consistency loss**:`J(θ) = E[||f(x,θ',η') - f(x,θ,η)||²]`
- MSE is used as the consistency loss function

#### **Experimental Results**

- **SVHN dataset**: 4.35% error rate under 250 labels
- **CIFAR-10 dataset**: improved from 10.55% to 6.28% under 4000 labels
- **ImageNet dataset**: improved from 35.24% to 9.11% under 10% labels

### **UA-MT (Uncertainty-Aware Mean Teacher)**

> **Paper link**: [Uncertainty aware multi-view co-training for semi-supervised medical image segmentation](https://arxiv.org/abs/1907.07034)

> **GitHub repository link**: [UA-MT](https://github.com/yulequan/UA-MT)

> **Publication year**: 2019 MICCAI

#### **Core innovation**

Introduces an **uncertainty-guided mechanism** based on Mean Teacher to calculate consistency loss only for high-confidence predictions:

- Uses **Monte Carlo Dropout** for uncertainty estimation
- Prediction entropy as uncertainty metric:`u = -Σc μc log μc`
- Dynamic threshold scheduling strategy, gradually learning from certain cases to uncertain cases

#### **Key Technologies**

```
# Uncertainty-guided consistency loss
Lc = [Σv I(uv < H) ||f'v - fv||²] / [Σv I(uv < H)]
```

#### **Experimental Results**

**Left atrial segmentation dataset (16 labeled + 64 unlabeled)**:

- Dice: 88.88%, Jaccard: 80.21%
- Significantly improved compared to supervised methods, approaching fully supervised performance

### **DTC (Dual-Task Consistency)**

> **Paper link**: [Deep Transform Consistency for Semi-supervised Medical Image Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/17066)

> **GitHub repository link**: [DTC](https://github.com/HiLab-git/DTC)

> **Publication year**: 2021 AAAI

#### **Core innovation**

Construct **task-level regularization** to establish consistency constraints between pixel-level classification tasks and geometry-aware level set regression tasks:

- **Task 1**: Traditional pixel-level segmentation
- **Task 2**: Level set function regression (capturing geometry and distance information)
- Use differentiable task transformation layers to connect the two tasks

#### **Technical Features**

- High efficiency: no need for multiple forward passes
- Fewer parameters: fewer parameters compared to complex multi-network architectures
- Strong generalizability: scalable to other task combinations

#### **Experimental Results**

**Pancreas segmentation (12 labeled + 50 unlabeled)**:

- Dice: 78.27%, Jaccard: 64.75%
- Training time of only 2.5 hours, significantly faster than other methods

### **UPC (Uncertainty-aware Pseudo-label and Consistency)**

> **Paper link**: [Uncertainty-aware Pseudo-label and Consistency for Semi-supervised Medical Image Segmentation](https://www.sciencedirect.com/science/article/pii/S1746809422006577)

> **GitHub repository link**: [UPC](https://github.com/AIforMS/UPC-Pytorch)

> **Publication year**: 2023 CBM

#### **Core innovation**

First to effectively combine **consistency regularization** and **pseudo-label learning** in medical image segmentation:

- Using KL divergence as an uncertainty metric
- Dynamic uncertainty mechanism to avoid the limitations of fixed thresholds
- End-to-end training framework

#### **Technical advantages**

- Computational efficiency: Training time reduced from 3.14 hours to 1.27 hours compared to the Monte Carlo Dropout method
- Adaptability: Dynamic threshold mechanism adapts to different prediction qualities
- Excellent performance: Outstanding performance in scenarios with scarce labeled data

#### **Experimental Results**

**Left Atrium Segmentation Dataset**:

- Dice: 89.65%, 95HD: 6.71 voxel
- Dice still reaches 84.37% with very few labeled data (4 labeled + 76 unlabeled)

### **MCF (Mutual Correction Framework)**

> **Paper Link**: [MCF: Mutual Correction Framework for Semi-Supervised Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MCF_Mutual_Correction_Framework_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.pdf)

> **GitHub repository link**: [MCF](https://github.com/WYC-321/MCF)

> **Publication year**: 2023 CVPR

#### **Core innovation**

Specifically addresses the problem of model cognitive bias in semi-supervised learning:

- **Heterogeneous sub-network design**: VNet + 3D-ResVNet, with different structures and independent parameters
- **Contrastive Difference Review (CDR)**: Uses XOR operations to find inconsistent areas and correct them
- **Dynamic Competitive Pseudo Label Generation (DCPLG)**: Real-time performance evaluation and dynamic selection of the best pseudo label generator

#### **Technical Highlights**

```
# Contrast Difference Review
M_diff = BINA(Ŷ^L_A) ⊕ BINA(Ŷ^L_B)  # XOR operation
L_rec = MSE(Ŷ^L_diff, Y^L_diff)       # Correction loss
```

#### **Experimental Results**

**Left atrial segmentation dataset**:

- Dice: 88.71%, 95HD: 6.32 voxel
- The CDR module can be integrated into other methods as a “free lunch”

### **BCP (Bidirectional Copy-Paste)**

> **Paper link**: [Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation](https://arxiv.org/pdf/2305.00673)

> **GitHub repository link**: [BCP](https://github.com/DeepMed-Lab-ECNU/BCP)

 

> **Publication year**: 2023 CVPR

#### **Core innovation**

Solves the **experience distribution mismatch** between labeled and unlabeled data:

- **Bidirectional mixing strategy**: inward mixing (unlabeled foreground + labeled background) + outward mixing (labeled foreground + unlabeled background)
- **Zero-centered mask design**: sufficient foreground-background interaction and optimal semantic preservation
- **Three-stage training**: pre-training → initialization → self-training

#### **Technical advantages**

- Distribution Alignment: Effectively alleviates the distribution mismatch between labeled and unlabeled data
- Simple and Efficient: No additional parameters required, computational overhead is the same as the baseline
- Broad Applicability: Effective across different modalities and organ segmentation tasks

#### **Experimental Results**

**Heart Segmentation (ACDC Dataset, 5% labeled)**:

- Dice: 87.59%, an improvement of 21.76% over the baseline
- Demonstrates significant advantages under extremely limited labeled data

### **CML (Cross-View Mutual Learning)**

> **Paper link**: [Cross-View Mutual Learning for Semi-Supervised Medical Image Segmentation](https://openreview.net/pdf/98a9ec17dea3e4d96eb45416cca53a8364aa93b1.pdf)

> **GitHub repository link**: [CML](https://github.com/SongwuJob/CML)

> **Publication year**: 2024 ACM MM

#### **Core innovation**

From “consistency learning” to “consistency + complementarity” collaborative learning:

- **Conflict feature learning (CFL)**: Encourages two subnetworks to learn different features from the same input
- **Cross-view mutual learning**: Constructing heterogeneous supervision signals through CutMix
- **Heterogeneous supervision signals**: Pseudo labels that mix one's own perspective and the other's perspective

#### **Core technology**

```
# Conflict feature learning
L_dis = 1 + (f_i · f̄_(1-i))/(||f_i|| × ||f̄_(1-i)||)
```

#### **Experimental Results**

**Left Atrial Segmentation Dataset (10% labeled)**:

- Dice: 90.36%, 95HD: 6.06 voxel
- Achieved SOTA results on three different modality datasets

### **AC-MT (Ambiguity-Consensus Mean Teacher)**

> **Paper Link**: [AC-MT: Ambiguity-Consensus Mean Teacher](https://www.sciencedirect.com/science/article/pii/S1361841523001408)

> **GitHub Repository Link**: [AC-MT](https://github.com/lemoshu/AC-MT)

 

> **Publication Year**: 2023 Medical Image Analysis

#### **Core Innovation**

Focus on **ambiguous but information-rich regions** for consistency learning:

- **Four ambiguous target selection strategies**: high Softmax entropy, high model uncertainty, prototype-guided label noise identification, and class-conditional systematic label noise identification
- **Adaptive target selection**: dynamically identify ambiguous but information-rich voxels
- **Plug-and-play design**: no need to modify the backbone network or introduce additional parameters

#### **Experimental Results**

**Left Atrium Segmentation (10% labeled data)**:

- Dice coefficient improved by 5.8% (84.25% → 89.12%)
- Achieved 81.18% Dice even under extremely scarce labeling (2.5%)

### **DyCON (Dynamic Uncertainty-aware Consistency)**

> **Paper Link**: [Dynamic Uncertainty-aware Consistency and Contrastive Learning](https://arxiv.org/pdf/2504.04566)

> **GitHub repository link**: [DyCON](https://github.com/KU-CVML/DyCON)

> **Publication year**: 2025

#### **Core innovations**

Handling high uncertainty caused by category imbalance and pathological changes:

- **Uncertainty-aware Consistency Loss (UnCL)**: Dynamically adjust voxel contributions to retain information in high uncertainty regions
- **Focused Entropy Contrast Loss (FeCL)**: Dual focus mechanism to handle category imbalance
- **Adaptive β strategy**: Dynamically adjust uncertainty handling based on training progress

#### **Technical Highlights**

```
# Uncertainty Consistency Loss
L_UnCL = (1/N) Σ [L(p^s_i, p^t_i) / (exp(β·H_s) + exp(β·H_t))] + (β/N) Σ [H_s + H_t]
```

#### **Experimental Results**

**ISLES-2022 Dataset (5% labeled)**:

- Dice: 61.48%, HD95: 17.61 voxel
- Excellent performance on highly challenging tasks such as stroke lesions

### **V-Net**

> **Paper link**: [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)

> **GitHub repository link**: [V-Net](https://github.com/mattmacy/vnet.pytorch)

> **Year of publication**: 2016 3DV

#### **Core innovation**

The first end-to-end fully convolutional neural network for 3D medical image segmentation:

- **3D volume convolution**: directly processes 3D volume data and maintains spatial continuity
- **Residual learning architecture**: learns residual functions at each stage to accelerate convergence
- **Dice loss function**: automatically handles class imbalance and directly optimizes segmentation quality metrics
- **V-shaped architecture**: encoder-decoder structure with skip connections

#### **Technical significance**

- Established the U-shaped architecture paradigm for medical image segmentation
- Dice loss became the standard for medical segmentation
- Laid the foundation for subsequent 3D segmentation networks

#### **Experimental results**

**PROMISE2012 prostate segmentation**:

- Dice: 0.869±0.033, Hausdorff distance: 5.71±1.20 mm
- Processing speed of approximately 1 second per volume, significantly faster than traditional methods

## **🔍 Method comparison and analysis**

### **Performance comparison table**

**Left atrial segmentation dataset (standard comparison settings)**

Method Annotated Data Dice (%) Jaccard (%) 95HD (voxels) ASD (voxels)

V-Net (supervised) 16 85.06 74.54 17.38 4.54

MT 16+64 85.89 76.58 12.63 3.44

UA-MT 16+64 88.88 80.21 7.32 2.26

DTC 16+64 89.42 80.98 7.32 2.10

UPC 16+64 89.65 81.36 6.71 2.15

MCF 16+64 88.71 80.41 6.32 1.90

CML 16+64 90.36 - 6.06 1.68

AC-MT 16+64 89.12 80.46 - -

### **Technical Evolution**

### **Method Classification**

#### **🎯 Classification by Core Technology**

**Consistency Regularization Class**:

- Mean Teacher, UA-MT, AC-MT, DyCON

**Multi-Task Learning Class**:

- DTC

**Pseudo-Label Combination Class**:

- UPC

**Multi-Model Collaboration Class**:

- MCF, CML

**Data Augmentation-Driven Class**:

- BCP

#### **🏆 Classification by Innovation Points**

**Uncertainty Handling**:

- UA-MT (filtering), UPC (dynamic), AC-MT (fuzzy region), DyCON (retain + utilize)

**Cognitive Bias Resolution**:

- MCF (heterogeneous networks), CML (complementary learning)

**Distribution Mismatch**:

- BCP (bidirectional copy-paste)

**Task-level constraints**:

- DTC (level set regression)

## **📊 Application datasets**

### **Introduction to commonly used datasets**

Dataset Modality Task Number of samples Features

**LA (Left Atrium)** MRI Left atrial segmentation 100 3D scans MICCAI 2018 challenge data

**ACDC** MRI Heart segmentation 150 patients Multi-class heart structure segmentation

**BraTS-2019** MRI Brain Tumor Segmentation 335 patients Multimodal brain tumor segmentation

**Pancreas-NIH** CT Pancreas Segmentation 82 CT scans Challenging small organ segmentation

**PROMISE2012** MRI Prostate Segmentation 50 training samples V-Net original validation dataset

**ISLES-2022** MRI Stroke Lesion Segmentation - High category imbalance scenario

Pancreas Dataset Introduction: The dataset has two versions. The initial version contains 82 CT scans (used by DyCON, etc.), while the updated version contains only 80 CT scans (used by BCP, etc.), with two problematic samples removed. We recommends using the latest version (80 CT scans). Here, i provides a Baidu Netdisk link [Baidu Netdisk](https://pan.baidu.com/s/50pwn9POkifuOXqrRKTPUxQ)

## **🚀 Usage Recommendations**

### **Method Selection Guide**

**Basic Introduction**:

- Start with **Mean Teacher** to understand the basic principles of the teacher-student framework
- Use **V-Net** as the backbone network for 3D medical image segmentation

**Uncertainty Handling**:

- **Adequate labeled data**: Choose UA-MT
- **Scarce labeled data**: Choose AC-MT
- **Extreme category imbalance**: Choose DyCON

**Performance Optimization**:

- **Need for optimal performance**: Try CML
- **Limited computing resources**: Choose UPC
- **Specific task adaptation**: Consider DTC

**Actual deployment**:

- **Stability priority**: MCF (cognitive bias correction)
- **Data augmentation requirements**: BCP
- **Multimodal data**: DyCON

### **Implementation recommendations**

1. **Environment configuration**: Most methods are implemented based on PyTorch and require CUDA support
2. **Data preprocessing**: Unify data formats and pay attention to normalization and enhancement strategies
3. **Hyperparameter Tuning**: Start with the parameters recommended in the paper and fine-tune them according to the specific dataset
4. **Evaluation Metrics**: Use standard metrics such as Dice coefficient, Jaccard index, and Hausdorff distance
5. **Training Strategy**: Pay attention to key parameters such as EMA updating, learning rate scheduling, and consistency weights

## **📖 Citation Format**

If these methods are helpful to your research, please consider citing the relevant papers:

```
@inproceedings{tarvainen2017mean,
title={Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results},
author={Tarvainen, Antti and Valpola, Harri},
booktitle={NIPS},
year={2017}
}

@inproceedings{yu2019uncertainty,
title={Uncertainty-aware self-ensembling model for semi-supervised 3D left atrium segmentation},
author={Yu, Lequan and Wang, Shujun and Li, Xiaomeng and Fu, Chi-Wing and Heng, Pheng-Ann},
booktitle={MICCAI},
  year={2019}
}

@inproceedings{luo2021semi,
title={Semi-supervised medical image segmentation through dual-task consistency},
author={Luo, Xiangde and Chen, Jieneng and Song, Tao and Wang, Guotai},
booktitle={AAAI},
```

  

```
year={2021}
}

# ... Other citation formats
```

## **💡 Future development directions**

### **Technical trends**

1. **Multimodal fusion**: Combining complementary information from different imaging modalities
2. **Lightweight design**: Reducing computational complexity to adapt to clinical deployment
3. **Enhanced interpretability**: Improving the interpretability of model decisions
4. **Domain adaptation**: Generalization ability across datasets and devices
5. **Active learning**: intelligently select the most valuable samples for annotation

### **Application extension**

- **Real-time segmentation**: surgical navigation and real-time diagnosis
- **Multi-organ joint segmentation**: simultaneous segmentation of organs throughout the body
- **Temporal data processing**: 4D medical image analysis
- **Cross-modal segmentation**: knowledge transfer between different imaging techniques

## **🤝 Contribution guidelines**

We welcome new methods, improvement suggestions, or bug fixes for this repository!

### **How to contribute**

1. **Fork** this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m ‘Add some AmazingFeature’`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a **Pull Request**

### **New Method Addition Format**

Please add new methods in the following format:

```
### Method Name

> **Paper Link**: [Title](link)
> **GitHub Repository Link**: [Title](link)
> **Publication Year**: Year Conference/Journal

#### Core Innovation
- Main Technical Innovations

#### Experimental Results
- Key performance data

```

## **📄 License**

This project is licensed under the MIT License - see the LICENSE file for details.

## **📞 Contact**

If you have any questions or suggestions, please contact us via the following methods:

- 📧 Email: [Zihao](https://www.deepl.com/zh/zihao3351@gmail.com)
- 🐛 Issues: [GitHub Issues](https://github.com/unumbrela/Semi-Supervised-Medical-Segmentation-Collection/issues)

**⭐ If this repository is helpful to you, please give it a Star to show your support!**