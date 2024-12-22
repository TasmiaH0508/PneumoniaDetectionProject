# Goal
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

Feature-wise variance maintained | 0.02 | 0.025 | 0.03 | 0.035 | 0.04 | 1 (i.e. No PCA applied) |
--- |----|-------|------|-------|------|-------------------------|
Percentage reduction in features | 16 | 27    | 42   | 60    | 73   |                         |

### Results Evaluation

### SVM

Due to the large number of features, a linear kernel was used and a high accuracy of at least 92% was achieved.

Feature-wise variance maintained | 0.02 | 0.025 | 0.03  | 0.035 | 0.04  | No PCA Applied|
--- |------|-------|-------|-------|-------|----|
Accuracy | 92   | 93.75 | 92.75 | 93    | 92.75 | |

### Neural Network