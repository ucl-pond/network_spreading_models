import numpy as np

# takes as input the output of a model (regions x timepoints) and the target data (regions)
# finds the best model timepoint and returns the index of that timepoint, the model prediction, 
# and the MSE between the model prediction and the target data

def mysse(xx, yy):
    '''returns sum-of-squared errors between inputs xx and yy'''
    if xx.ndim == 1:
        xx = np.expand_dims(xx, 1)
    if yy.ndim == 1:
        yy = np.expand_dims(yy, 1)

    sse = np.sum((xx-yy)**2, axis=0)
    return sse

def find_optimal_timepoint(model_output, target_data):
    n_regions, n_t = np.shape(model_output)
    SSE = mysse(model_output, target_data)

    # find the timepoint with the lowest SSE   
    min_idx = np.argmin(SSE)
    prediction = model_output[:,min_idx]

    return min_idx, prediction, SSE[min_idx]

