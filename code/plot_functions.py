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
  
def plot_times_series(data_frame, cmap):
    fig,ax = plt.subplots(figsize=(15, 5))
    for x in range(0, len(data_frame.columns), 1):
        cmap = plt.get_cmap(cmap)
        slicedCM = cmap(np.linspace(0, 1, len(df.columns))) 
        ax.plot(data_frame.index, data_frame.iloc[:,x], color=slicedCM[x])
        plt.xlabel("Time")
        plt.ylabel("Predicted tau accumulation")
        fig.show()
