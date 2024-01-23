# functions for plotting output from the models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_seed_correlations(data_frame, cmap):
    figure, ax = plt.subplots(figsize=(15, 10))
    df1 = df[df['r'].notnull()] # drop NaNs
    df2 = df1.sort_values(by="r") # sort by r correlation value
    y = df2.iloc[:,1] # set y as r
    my_cmap = plt.get_cmap(cmap)
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    ax.bar(df2.iloc[:,2], y, color=my_cmap(rescale(y)))
    plt.xticks(rotation=90)
    plt.xlabel("Seed Region")
    plt.ylabel("Correlation Pearson's r")
    plt.show()
  
def plot_times_series(model_output, t, cmap):
    n_seeds = model_output.shape[0]
    fig,ax = plt.subplots(figsize=(15, 5))
    for x in range(n_seeds):
        cmap = plt.get_cmap(cmap)
        slicedCM = cmap(np.linspace(0, 1, len(data_frame.columns))) 
        ax.plot(data_frame.index, data_frame.iloc[:,x], color=slicedCM[x])
        plt.xlabel("Time",fontsize=20)
        plt.ylabel("Predicted tau accumulation",fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        fig.show()
