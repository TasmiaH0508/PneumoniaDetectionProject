import torch

def get_accuracy(actual_labels, predicted_labels):
    ''''
    Takes in actual labels, which is a torch array and predicted labels, which is a torch array
    
    Note: Labels are 0 or 1 (Binary Classification)
    
    Returns percentage of correctly classified samples
    '''
    total_labels = actual_labels.shape[0]

    actual_labels = torch.reshape(actual_labels, (total_labels,))
    predicted_labels = torch.reshape(predicted_labels, (total_labels,))

    number_of_correctly_classified_points = torch.where(actual_labels == predicted_labels)[0].shape[0]

    return number_of_correctly_classified_points / total_labels * 100

def get_precision():
    return 0

def get_num_true_positives(actual_labels, predicted_labels):
    '''''''''
    Returns the number of points correctly classified as 1 
    '''''''''
    actual_labels = torch.reshape(actual_labels, (actual_labels.shape[0],))
    predicted_labels = torch.reshape(predicted_labels, (predicted_labels.shape[0],))
    indices_where_pred_and_actual_match = torch.where(actual_labels == predicted_labels)[0]
    predicted_labels = predicted_labels[indices_where_pred_and_actual_match]
    num_true_positives = torch.where(predicted_labels == 1)[0].shape[0]
    return num_true_positives

def get_num_false_positives(actual_labels, predicted_labels):
    '''''''''
    Returns the number of points wrongly classified as 1
    '''
    actual_labels = torch.reshape(actual_labels, (actual_labels.shape[0],))
    predicted_labels = torch.reshape(predicted_labels, (predicted_labels.shape[0],))
    indices_of_wrongly_classified_points = torch.where(actual_labels != predicted_labels)[0]
    wrongly_classified_pts = actual_labels[indices_of_wrongly_classified_points]
    num_false_positives = torch.where(wrongly_classified_pts == 0)[0].shape[0]
    return num_false_positives

def get_num_false_negatives(actual_labels, predicted_labels):
    ''''''''''
    Returns the number of points wrongly classified as 0
    '''''''''
    actual_labels = torch.reshape(actual_labels, (actual_labels.shape[0],))
    predicted_labels = torch.reshape(predicted_labels, (predicted_labels.shape[0],))
    indices_where_pred_and_actual_do_not_match = torch.where(actual_labels != predicted_labels)[0]
    predicted_labels = predicted_labels[indices_where_pred_and_actual_do_not_match]
    num_true_negatives = torch.where(predicted_labels == 0)[0].shape[0]
    return num_true_negatives

def get_precision(actual_labels, predicted_labels):
    tp = get_num_true_positives(actual_labels, predicted_labels)
    fp = get_num_false_positives(actual_labels, predicted_labels)
    return tp / (tp + fp)

def get_recall(actual_labels, predicted_labels):
    tp = get_precision(actual_labels, predicted_labels)
    fn = get_num_false_positives(actual_labels, predicted_labels)
    return tp / (tp + fn)

actual_labels = torch.asarray([1, 0, 0])
predicted_labels = torch.asarray([0, 0, 1])

print(get_num_false_negatives(actual_labels, predicted_labels))