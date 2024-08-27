# SimCLR-SAR-ATR-Project
This is a SAR ATR project using the contrastive learning model SimCLR. This is a university project based on a paper:
[Paper Link](https://www.sciencedirect.com/science/article/pii/S1877050922014697)

[![Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ysjMt-EeY1dCM2WFBuefTr4kaHS6_pmC?usp=sharing)

## Introduction
<p>In the field of automatic target recognition using synthetic aperture radar (SAR) images, deep learning methods face significant challenges when dealing with small sample sizes. Although deep learning has shown great potential in SAR image recognition, it often struggles with issues like overfitting and gradient explosion when the available data is limited. To address these challenges, this study trains a deep neural network on unlabeled SAR image data using self-supervised contrastive learning to create a network capable of extracting deeper features and achieving better representation. The pre-trained weights can then be transferred to small sample SAR data, in order to fine-tune the model to reduce overfitting in small sample SAR image recognition.
</p>
<p align="center">
  <img src="/media/TargetRecognitionFromSARImagesUsingDeepLearningExample_01.png" alt="SAR-ATR" title="SAR-ATR" width="500"/>
</p>

## Related Works and Challenges
- Deep Learning in SAR Image Classification: Previous works have extensively explored the application of deep learning techniques, particularly convolutional neural networks (CNNs), to synthetic aperture radar (SAR) image classification. These methods have demonstrated significant improvements in accuracy and robustness compared to traditional machine learning techniques. However, deep learning models typically require large amounts of labeled data to achieve optimal performance, which is often a limitation in SAR datasets.

- Transfer Learning and Pre-training: To mitigate the challenge of limited labeled SAR data, several studies have employed transfer learning and pre-training strategies. Transfer learning involves pre-training a model on a large, related dataset and then fine-tuning it on a smaller, specific dataset. This approach leverages the feature extraction capabilities learned from the larger dataset, improving performance on the smaller, target dataset. Prior works have shown that transfer learning can be particularly effective for SAR image classification when the source and target domains are similar.

- Self-Supervised Learning: More recent studies have introduced self-supervised learning techniques to pre-train models using unlabeled data. In the context of SAR image analysis, self-supervised learning methods like contrastive learning have been used to learn robust feature representations without requiring labeled data. These methods have been shown to enhance model performance on downstream tasks by pre-training on large amounts of unlabeled SAR data and then fine-tuning on smaller labeled datasets.

- Challenges with Small Sample Sizes: A recurring theme in related works is the difficulty of achieving high accuracy with deep learning models when working with small sample sizes, which is common in SAR image datasets. Overfitting and poor generalization are common issues due to the limited amount of training data. To address these problems, various techniques such as data augmentation, regularization, and transfer learning have been proposed and evaluated in previous studies.
## Contrastive Learning
<p>Contrastive learning is a type of self-supervised learning technique used to train models to learn useful representations of data without requiring labeled samples. The main idea behind contrastive learning is to learn representations by distinguishing between similar (positive) and dissimilar (negative) data pairs.<br>

<p align="center">
  <img src="/media/contrastive_standard.png" alt="Contrastive Learning" title="Contrastive Learning" width="500"/>
</p>

### Key Concepts of Contrastive Learning:
- ####  Representation Learning: 
The goal of contrastive learning is to learn a feature space where similar data points are closer together, and dissimilar points are further apart. This helps the model understand the underlying structure of the data.

- #### Positive and Negative Pairs:
Positive pairs are composed of data points that are considered similar. For instance, two different augmentations of the same image.
Negative pairs consist of data points that are different from each other. For example, augmentations of different images.
- #### Contrastive Loss:
The training objective of contrastive learning is to minimize the distance between the representations of positive pairs while maximizing the distance between negative pairs.
A commonly used loss function in contrastive learning is the contrastive loss or InfoNCE loss, which encourages the model to increase the similarity of positive pairs and decrease the similarity of negative pairs.
- #### Augmentations:
Data augmentations are crucial in contrastive learning. For images, this could involve random cropping, flipping, color jittering, etc. The goal of augmentations is to create diverse versions of the same image to help the model learn invariant features.

- #### Self-Supervised Learning:
Contrastive learning falls under self-supervised learning because it does not require labeled data. Instead, it uses data augmentations to generate pseudo-labels (positive and negative pairs) to train the model.</p>

## Proposed Solution: SimCLR
SimCLR (Simple Framework for Contrastive Learning of Visual Representations) is a popular self-supervised learning framework developed by researchers at Google Research for learning visual representations from unlabeled data. SimCLR leverages contrastive learning techniques to train models to distinguish between similar and dissimilar data points, effectively learning useful features without the need for manually labeled datasets.
<p align="center">
  <img src="/media/1_GuoSK8ghNX11JUlq-j0LYw.png" alt="SimCLR" title="SimCLR" width="500"/>
</p>

### Self-Supervised Learning:

SimCLR is a self-supervised learning method, meaning it does not rely on labeled data. Instead, it creates its own "labels" using transformations of the data.
This approach is useful for scenarios where obtaining labeled data is expensive or impractical such as this case.

### Contrastive Learning:

At the core of SimCLR is contrastive learning, a technique where the model learns to differentiate between similar (positive) and dissimilar (negative) pairs of images.
In SimCLR, positive pairs are created by applying two different random data augmentations to the same image. Negative pairs consist of augmentations from different images.

### Data Augmentation:

Data augmentation is crucial in SimCLR. Each image in the dataset is augmented twice using random transformations like cropping, color jittering, flipping, and blurring. This creates two different views (augmentations) of the same image, forming a positive pair.
These augmentations help the model learn invariant features — features that remain consistent across different transformations of the same image.

### Neural Network Encoder:

SimCLR uses a deep neural network (typically a ResNet) as an encoder to transform input images into a lower-dimensional representation (feature vector).
The encoder extracts meaningful features from the images, which are then used to distinguish between positive and negative pairs.

### Projection Head:

After the encoder, SimCLR uses a small neural network called the "projection head" that maps the encoded representations to a space where contrastive loss is applied.
This projection head is typically a multi-layer perceptron (MLP) with one or two layers. The output of the projection head is the space where the similarity between pairs is calculated.
This step helps improve the quality of the learned representations by focusing on the parts of the representation that are most useful for distinguishing between positive and negative pairs.

### Contrastive Loss (NT-Xent Loss):

SimCLR uses the normalized temperature-scaled cross-entropy loss (NT-Xent loss) as its contrastive loss function.
For each image in a batch, the model tries to maximize the agreement between positive pairs (two augmented versions of the same image) while minimizing the agreement between negative pairs (augmented versions of different images).
This loss function pushes similar images closer in the feature space and dissimilar images further apart.

### Training:

The model is trained using a large batch size to ensure a sufficient number of negative pairs for each positive pair, which is critical for effective contrastive learning.
During training, the encoder and projection head learn to create a feature space where similar images are grouped closely together, and different images are spread apart.

### Representation Learning:

After training, the projection head is typically discarded, and the encoder is used as a pre-trained model for various downstream tasks such as image classification, object detection, or segmentation.
The representations learned through SimCLR are general and can be fine-tuned on smaller labeled datasets for specific tasks.

## Datasets
### [MSTAR 8 Classes](https://www.kaggle.com/datasets/atreyamajumdar/mstar-dataset-8-classes)
The MSTAR (Moving and Stationary Target Acquisition and Recognition) dataset is a publicly available synthetic aperture radar (SAR) dataset that contains high-resolution SAR images of various military targets and civilian vehicles. It is widely used in research for automatic target recognition (ATR) and SAR image processing.
### [SARScope](https://www.kaggle.com/datasets/kailaspsudheer/sarscope-unveiling-the-maritime-landscape)
This dataset is designed for researchers and developers interested in Synthetic Aperture Radar (SAR) ship detection and instance segmentation. It combines the advantages of both HRSID and OPEN-SSDD datasets, offering a diverse and robust collection of data ,taking the total images count to 6735.
