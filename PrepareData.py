def get_data_without_bias_and_label(data):
    ''''
    Removes the labels and bias cols from data(torch tensor)
    '''
    num_features = data.shape[1] - 1
    data = data[:, 1 : num_features]
    return data

def get_label(data):
    '''''
    Takes out the last column, which is the label col, from the data(torch tensor)
    '''
    num_features = data.shape[1] - 1
    label = data[:, num_features]
    return label