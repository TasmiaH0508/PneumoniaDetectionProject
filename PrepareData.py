def get_data_without_bias_and_label(data, has_bias=True, has_label=True):
    ''''
    Removes the labels from data(torch tensor)
    Bias col is removed if present, as indicated by has_bias
    Label col is removed if present, as indicated by has_label
    '''
    if has_label:
        num_features = data.shape[1] - 1
    else:
        num_features = data.shape[1]
    if has_bias:
        data = data[:, 1: num_features]
    else:
        data = data[:, 0: num_features]
    return data

def get_label(data):
    '''''
    Takes out the last column, which is the label col, from the data(torch tensor)
    '''
    num_features = data.shape[1] - 1
    label = data[:, num_features]
    return label