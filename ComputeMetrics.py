import torch
import numpy as np

def get_accuracy(actual_labels, predicted_labels):
    ''''''''''
    Takes in actual labels, which is a torch array of shape: (number of samples, 1) and 
    predicted labels, which is a torch array of shape: (number of samples, 1)
    
    Note: Labels are 0 or 1 (Binary Classification)
    
    Returns percentage of correctly classified samples
    '''''
    total_labels = actual_labels.shape[0]

    actual_labels = torch.reshape(actual_labels, (total_labels,))
    predicted_labels = torch.reshape(predicted_labels, (total_labels,))

    number_of_correctly_classified_points = torch.where(actual_labels == predicted_labels)[0].shape[0]

    return number_of_correctly_classified_points / total_labels * 100