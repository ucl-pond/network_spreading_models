import numpy as np
from scipy.stats import spearmanr
from scipy.stats import entropy

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

def spearmans_rank(xx, yy):
    num_t = np.shape(xx)[1] ## assuming xx is model output
    sp = [np.nan_to_num(spearmanr(yy, xx[:,i]).statistic) for i in range(num_t)]
    
    return 1 - np.array(sp)

def KL_divergence(xx, yy):
    if xx.ndim == 1:
        xx = np.expand_dims(xx, 1)
    if yy.ndim == 1:
        yy = np.expand_dims(yy, 1)

    xx = xx/np.sum(xx, axis=0)
    yy = yy/np.sum(yy, axis=0)

    xx = xx + 0.00001
    yy = yy + 0.00001
    return entropy(xx, yy) + entropy(yy, xx)

def pearsons_r(xx, yy):
    '''returns 1 - Pearson's correlation between inputs xx and yy'''
    num_t = np.shape(xx)[1] ## assuming xx is model output
    R = [np.corrcoef(yy, xx[:,i])[0,1] for i in range(num_t)]
    
    return 1 - np.array(R)



def find_optimal_timepoint(model_output,
                           target_data,
                           objective_function=mysse):
    
    objective = objective_function(model_output, target_data)
    
    # find the timepoint with the lowest objective
    min_idx = np.argmin(objective)
    prediction = model_output[:,min_idx]

    return min_idx, prediction, objective[min_idx]
