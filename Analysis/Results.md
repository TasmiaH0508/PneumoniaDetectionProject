# Goal of Project
Train an image classifier that can distinguish x-ray photographs of patients with pneumonia 
and those without.

### Models explored
- SVM
- Neural Network
- CNN
- Standard Logistic Regression

### Processing the Data

The method of processing images for SVM and Neural Network was different from CNN.

For SVM and Neural Network, the images where processed such that every image was represented as a list.
Since every x-ray photograph was 65536 pixels and there were about 3600 images, the data matrix was too 
large, motivating the use of PCA.

Before PCA was applied, about 10%(or 200) of the data points were randomly selected from each data group(Pneumonia
and non-pneumonia). Normalisation was then applied to the test set. After applying PCA to the training data(and 
reconstructing the data), features showing very little variance were removed from both the test and training data. Since
the training set was already normalised, the test set was normalised separately on its own.

The labels were then added such that if pneumonia was present, a label of 1 was given and a label of 0 to those without 
Pneumonia.

Table showing the percentage reduction in features by varying the variance of features:

Feature-wise variance maintained | 0.02 | 0.025 | 0.03 | 0.035 | 0.04 |
--- |----|-------|------|-------|------|
Percentage reduction in features | 16 | 27    | 42   | 60    | 73   |

### Results Evaluation

### SVM

Two kinds of kernels were used: Gaussian and Linear.

For the linear kernel and regularisation param c = 1.0:

Feature-wise variance maintained | 0.02 | 0.025 | 0.03 | 0.035 | 0.04 |
--- |----|-------|------|-------|------|
Percentage accuracy | 91.2 | 90.9    | 90.4   | 89.6    | 88.2   |

For the Gaussian kernel and regularisation param c = 1.0:

Feature-wise variance maintained | 0.02 | 0.025 | 0.03 | 0.035 | 0.04 |
--- |------|-------|------|-------|------|
Percentage accuracy | 91.8 | 91.7  | 91.6 | 91.1  | 90.0 |

#todo: maybe use confusion matrix do deduce which is the best model based on recall/precision

### Neural Network

### CNN

### Standard Logistic Regression