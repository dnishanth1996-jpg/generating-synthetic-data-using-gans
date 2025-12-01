# Generating Tabular Synthetic Data

#### Table of Contents

[TOC]

# Executive Summary

##### Problem context

Advanced analytics is transforming all industries and is inherently data hungry. In health care, data privacy rules detract from data sharing for collaboration. Synthetic data, which retains the original characteristics and is model compatible, can make data sharing easy and enable analytics for health care data.

##### Need for change

Conventionally, statistical methods have been used, but with limited success. Current de-identification techniques are not sufficient to mitigate re-identification risks. Emerging technologies in Deep Learning such as GANs are very promising to solve this problem.

##### Key Question

How can you certify that the generated data is as similar and as useful as original data for the intended uses?

##### Proposed Solution

The proposed solution involves generating Synthetic Data using Generative Adversarial Networks (GANs) with the help of available sources such as TGAN and CTGAN.

The project includes modules designed to test the generated synthetic data against the original datasets in the following three areas:

- Statistical Similarity: Standardized modules to check if the generated datasets are similar to the original dataset.
- Privacy Risk: Standardized metrics to check if generated synthetic data protects the privacy of all data points in the original dataset.
- Model Compatibility: Comparison of Machine Learning techniques on original and synthetic datasets.

##### Results:

- Statistical Similarity: The generated datasets were similar to each other when compared using PCA and Auto-Encoders.
- Privacy module: The generated Privacy at Risk (PaR) metric and modules help identify columns which are at risk of exposing the privacy of data points from original data. The generated datasets using TGAN and CTGAN had sufficiently high privacy scores and protected the privacy of original data points.
- Model Compatibility: The synthetic data has comparable model performance for both classification and regression problems.

# Introduction

Healthcare organizations deal with sensitive data that contains Personal Identifiable Information (PII). The healthcare industry is particularly sensitive as Patient Identifiable Information data is strictly regulated by the Health Insurance Portability and Accountability Act (HIPAA) of 1996. Firms need to keep customer data secure while leveraging it to innovate research and drive growth. However, current data sharing practices to ensure de-identification have resulted in long wait times for data access. This has proved to be a hindrance to fast innovation. The goal is to reduce the time for data access and enable innovation while protecting the information of patients. The key question to answer here is:

"How can we safely and efficiently share healthcare data that is useful?"

##### Complication

The key questions involve the inherent trade-off between safety and efficiency. With the inception of big data, efficiency in the data sharing process is of paramount importance. Availability and accessibility of data ensure rapid prototyping and lay down the path for quick innovation in the healthcare industry. Efficient data sharing also unlocks the full potential of analytics and data sciences through use cases like the diagnosis of cancer, predicting response for drug therapy, vaccine developments, and drug discovery through bioinformatics. Apart from medical innovation, efficient data sharing helps to bridge the shortcomings in the healthcare system through salesforce effectiveness, managing supply chains, and improving patient engagement. While efficient data sharing is crucial, the safety of patient data cannot be ignored. Existing regulations like HIPAA and recent privacy laws like the California Consumer Privacy Act are focused on maintaining the privacy of sensitive information. More advanced attacks are being organized by hackers aimed at accessing personal information. As per reports on cost data breaches, the cost per record is approximately $150. But the goodwill and trust lost by companies cannot be quantified. So, the balance between data sharing and privacy is vital.

# History of the Project

Existing de-identification techniques involve two main techniques: 1) Anonymization Techniques and 2) Differential Privacy. Almost every firm relies on these techniques to deal with sensitive information in PII data.

1. Anonymization techniques: These techniques try to remove the columns which contain sensitive information. Methods include deleting columns, masking elements, quasi-identifiers, k-anonymity, l-diversity, and t-closeness.

2. Differential privacy: This is a perturbation technique which adds noise to columns to introduce randomness to data and thus maintain privacy. It is a mechanism to help maximize the aggregate utility of databases ensuring high levels of privacy for the participants by striking a balance between utility and privacy.

However, these techniques are not always cutting edge when it comes to maintaining privacy and data sharing. Research has shown that a high percentage of individuals in sampled datasets can be correctly re-identified using a limited number of demographic attributes. This challenges the technical and legal adequacy of the de-identification release-and-forget model.

**Proposition**

Currently, the field of AI which is being given a lot of importance is Deep Learning. It addresses the critical aspect of data science through universality theorem and representation learning. Generative modeling has seen a rise in popularity. In particular, a model called Generative Adversarial Networks or GANs shows promise in producing realistic samples. While these are state-of-the-art deep learning models to generate new synthetic data, there are challenges to overcome.

- Neural Network is a cutting edge algorithm in the industry: Trained to solve one specific task, can it fit all use cases?
- Generate image using CNN architecture: Can we generate tables from relational databases?
- Generate fake images of human faces that look realistic: Would it balance the trade-off between maintaining utility and privacy of data?
- Requires high computational infrastructure like GPUs: How to implement GAN for big data?

# Why GANs?

## Introduction

A generative adversarial network (GAN) is a class of machine learning systems invented by Ian Goodfellow in 2014. GAN uses algorithmic architectures that use two neural networks, pitting one against the other (the "adversarial" component) in order to generate new, synthetic instances of data that can pass for real data.

GANs consist of two neural networks contesting with each other in a game. Given a training set, this technique learns to generate new data with the same statistics as the training set. The two Neural Networks are named the Generator and the Discriminator.

## GAN Working Overview

**Generator**
The generator is a neural network that models a transform function. It takes as input a simple random variable and must return, once trained, a random variable that follows the targeted distribution. The generator feeds actual data and generated data to the Discriminator. The generator starts with generating random noise and changes its outputs as per the Discriminator. If the Discriminator is successfully able to identify that the input is fake, then its weights are adjusted to reduce the error.

**Discriminator**
The Discriminator's job is to determine if the data fed by the generator is real or fake. The discriminator is first trained on real data so that it can identify it to acceptable accuracy. If the Discriminator is not trained properly, then it will not be accurately able to identify fake data, thus poorly training the Generator.

This is continued for multiple iterations until the discriminator can identify the real/fake data purely by chance only.

**Algorithm:**
- The generator randomly feeds real data mixed with generated fake data for the discriminator.
- To begin, in the first few iterations, the generator produces random noise which the discriminator is very good at detecting as fake.
- Every iteration the discriminator catches a generated image as fake, the generator readjusts its weights to improve itself, similar to the Gradient Descent algorithm.
- Over time, after multiple iterations, the generator becomes very good at producing data which can fool the discriminator and pass as real.
- Now, it is the discriminator's turn to improve its detection algorithm by adjusting its network weights.
- This game continues until a point where the discriminator is unable to distinguish a real image from fake and can only guess by chance.

## Existing Research in Synthetic Data Generation

### TGAN

This methodology has been created based on the work provided in the paper: "Synthesizing Tabular Data using Generative Adversarial Networks" and the associated Python package.

Generative adversarial networks (GANs) implicitly learn the probability distribution of a dataset and can draw samples from the distribution. Tabular GAN (TGAN) is a generative adversarial network which can generate tabular data by learning the distribution of existing training datasets. TGAN focuses on generating tabular data with mixed variable types (multinomial/discrete and continuous). To achieve this, it uses LSTM with attention in order to generate data column by column.

#### Data preparation

For a table containing discrete and continuous random variables, each row in the table is a sample from a probability distribution. The algorithms learn a generative model such that samples generated from this model can satisfy two conditions:
- A Machine Learning model using the Synthetic table achieves similar accuracy on the test table.
- Mutual information between an arbitrary pair of variables is similar.

**Numerical Variables**
For the model to learn the data effectively, a reversible transformation is applied. Numerical variables are converted into a scalar in the range (-1, 1) and a multinomial distribution. Gaussian Kernel density estimation is used to estimate the number of modes in the continuous variable. To sample values from these, a gaussian mixture model is used.

**Categorical Variables**
Categorical variables are converted to one-hot-encoding representation and noise is added to binary variables. In TGAN, the discriminator D tries to distinguish whether the data is from the real distribution, while the generator G generates synthetic data and tries to fool the discriminator. The algorithm uses a Long Short Term Memory (LSTM) as the generator and a Multi Layer Perceptron (MLP) as a discriminator.

#### Implementation

```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import tensorflow as tf
from tgan.model import TGANModel
from tgan.data import load_demo_data

def tgan_run(data, cont_columns):
    tgan = TGANModel(continuous_columns)
    return tgan.fit(data)

def tgan_samples(model, num_samples):
    return tgan.sample(100000)
```

### CTGAN

CTGAN is a GAN-based method to model tabular data distribution and sample rows from the distribution. CTGAN implements mode-specific normalization to overcome non-Gaussian and multimodal distributions. It uses a conditional generator and training-by-sampling to deal with imbalanced discrete columns.

**Several unique properties of tabular data challenge the design of a GAN model:**

- Mixed data types: Real-world tabular data consists of mixed types. GANs must apply both softmax and tanh on the output.
- Non-Gaussian distributions: Continuous values in tabular data are usually non-Gaussian where min-max transformation will lead to vanishing gradient problems.
- Multimodal distributions: Vanilla GANs struggle in modeling the multimodal distribution of continuous columns.
- Learning from sparse one-hot-encoded vectors: A trivial discriminator can simply distinguish real and fake data by checking the distribution's sparseness.
- Highly imbalanced categorical columns: Imbalanced data leads to insufficient training opportunities for minor classes.

When feeding data to the GAN algorithm, CTGAN samples so that all categories are correctly represented. The goal is to resample efficiently so that all categories from discrete attributes are sampled evenly during the training process.

#### Implementation

```python
import pandas as pd
import tensorflow as tf

from ctgan import load_demo
from ctgan import CTGANSynthesizer

data = load_demo()
discrete_columns = ['workclass','education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country', 'income']
ctgan = CTGANSynthesizer()
ctgan.fit(data, discrete_columns)
```

### Differentially Private GAN

One common issue in GAN methodologies is that the density of the learned generative distribution could concentrate on the training data points, meaning they can remember training samples. This is a concern when GANs are applied to private or sensitive data. Differentially Private GANs are achieved by adding carefully designed noise to gradients during the learning procedure. DPGAN focuses on preserving privacy during the training procedure instead of adding noise on the final parameters directly.

### PATE-GAN

PATE GAN modifies the existing GAN algorithm in a way that guarantees privacy. It consists of two generator blocks called the student block and teacher block. PATE GAN prevents the reconstruction of original data by breaking down the generator into three stages. The synthetic data is private with respect to the original data.

### G-PATE

G-PATE ensures differential privacy on the information flow from the discriminator to the generator. Compared to PATE-GAN, this approach improves the use of the privacy budget by applying it to the part of the model that actually needs to be released for data generation. The gradient aggregator sanitizes the information flow from the teacher discriminators to the student generator.

# Methodology

In order to validate the efficacy of GANs, this project utilizes a methodology for thorough evaluation of synthetic data.

## MIMIC - III Dataset

MIMIC-III (Medical Information Mart for Intensive Care Unit) is an open source database containing de-identified health data of 40,000+ patients admitted to the ICU.

- 26 tables: Comprehensive clinical dataset.
- 45,000+ patients: Includes details like Gender, Ethnicity, Marital Status.
- 65,000+ ICU admissions: Data from 2001 to 2012.
- PII data: Includes patient sensitive details like demographics, diagnosis, etc.

Two different use cases were created for this project: Length of Stay and Mortality Prediction.

## Use Case 1: Length of Stay

#### Goal
Predicting the length of stay in ICU using 4 out of 26 tables. This helps in predicting the tendency of the length of stay of a patient, which is beneficial for resource management and planning.

#### Methodology
1. Load csv files: Data from Admissions, Patients, ICUSTAYS, and Diagnosis_ICD.
2. Merge tables: Use SUBJECT_ID and HADM_ID to merge relevant information.
3. Prepare diagnosis dataset: Map ICD-9 codes into 18 different disease area categories.
4. Pivot up diagnosis dataset: Aggregate counts of diagnoses across categories.
5. Merge pivoted diagnosis dataset to the main dataset.

#### Final Data
- Level of data: Unique admission for each patient.
- Target Variables: 'LOS' (length of stay).
- Predictor variables: 18 columns of different diagnosis categories and descriptive variables like ADMISSION_TYPE, INSURANCE, etc.

## Use Case 2: Mortality Prediction

#### Goal
Predicting mortality based on the number of interactions between patient and hospital (lab tests, prescriptions, procedures). It serves as a classification problem to predict if a patient will expire during a hospital admission.

#### Methodology
1. Load csv files: Multiple tables including Patients, Admissions, CallOut, CPTEvents, etc.
2. Roll up tables: Aggregate counts of events at the SUBJECT_ID and HADM_ID level.
3. Merge tables: Use composite keys to create the final analytical data set.

# Synthetic Data Generation

Synthetic data can be tuned to add privacy without losing utility or exposing individual data points. As the data does not represent any real entity, the disclosure of sensitive private data is eliminated.

##### Configuring GPU for Tensorflow
To take advantage of GPU for faster training, the system must be equipped with a CUDA enabled GPU card with compatibility for CUDA 3.5 or higher.

##### Required Python Packages
```python
pip install tensorflow 
```

##### Software requirements
- NVIDIA GPU drivers
- CUDA Toolkit
- CUPTI
- cuDNN SDK

# Scalability Tests

Training time for TGAN and CTGAN algorithms is a significant part of the process. GANs involve two neural networks competing, which requires resources.

#### TGAN
- Categorical Data: Training time is mainly affected by the number of columns.
- Continuous Data: Increasing rows does not significantly increase training time compared to columns.

#### CT-GAN
- Categorical Data: Training time is affected by both rows and columns.
- Numerical Data: Takes much lower time to train than TGAN.

# Statistical Similarity

#### Goal
To calculate the similarity between two tables by evaluating how different the synthetic data is from the original data. Evaluation is done column-wise and table-wise.

#### Methodology

##### 1. Column-wise Similarity Evaluation
- KL-divergence: Measures distribution similarity between pair-columns.
- Cosine Similarity: Measures similarity between two non-zero vectors, useful when distributions have different lengths.

##### 2. Table-wise Similarity Evaluation
- Autoencoder: Data compression algorithm used to learn a representation of relationships among multiple features.
- PCA and t-SNE: Dimensionality reduction techniques for visualization. PCA is preferred for comparison due to its mathematical nature vs the probabilistic nature of t-SNE.
- Clustering Metric: Using k-means clustering to organize data into groups and comparing cluster centers between original and synthetic data.

# Privacy Risk Module

Healthcare firms are vulnerable to data breaches. Synthetic data can solve the problem, but there is a need to compare privacy risks.

#### Goal
Quantitatively measure privacy risk to understand the level of privacy achieved in a synthetic dataset.

##### Roadmap of Privacy Risk Measurement
1. Exact Matches: Scanning for exact clones in synthetic data.
2. Closest Matches: Looking at who is the closest match to original records.
3. Privacy at Risk (PaR): Leveraging the idea of confusion. If internal similarity is greater than external similarity, it is a low-risk situation.
4. PaR using Distance Metrics: Using Euclidean distance to find the closest resemblance.

##### Applications of PaR Module
1. Privacy Risk Removal: Identifying and removing synthetic data points that contribute to high risk.
2. Measuring Sensitivity of Columns: Understanding how much each column contributes to the overall privacy risk.

# Model Compatibility

#### Goal
Evaluate if models generated using synthetic data are compatible with original data.

#### Methodology
1. One hot encoding.
2. Split data into train and test.
3. Generate Synthetic Data.
4. Standardize variables.
5. Model building: Regression (Random Forest, XGBoost, KNN) and Classification (Logistic Regression, XGBoost, Neural Network).
6. Hyperparameter tuning.
7. Prediction and metric comparison.

# Further Research and Improvements

## Generating Larger Datasets from Small Samples
Synthetic data can be used to generate larger datasets for training deep learning models. Statistical similarity and privacy must be verified for scaled data.

## Primary Keys
Future implementations can account for the uniqueness of ID fields to preserve primary key consistency.

## Privacy Tuning
Understanding the balance between Utility and Privacy at Risk values is vital. Future research can focus on including PaR in the objective function of GANs.

## Model Efficiency
Optimizing code to parallelize operations can make the Synthetic Data workflow more scalable.

# Sources
1. Research on Differentially Private Generative Adversarial Networks.
2. Documentation on Differential Privacy and Data Anonymization techniques.
3. Technical blogs on Synthetic Data Generation and Dimensionality Reduction.

---

### Maintainer

**Nishanth Doddipalli**
Data Engineer
Email: dnishanth1996@gmail.com
LinkedIn: https://www.linkedin.com/in/nishanth-doddipalli/

Nishanth is a Data Engineer with over 7 years of experience in the healthcare and finance sectors. He specializes in building robust data pipelines and scalable data solutions using Python, SQL, and cloud platforms like AWS and Azure. This project focuses on the intersection of data engineering and privacy-preserving analytics for sensitive healthcare data. He actively maintains this repository to explore advanced synthetic data generation techniques and privacy metrics.