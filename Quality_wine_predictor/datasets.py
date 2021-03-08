from os import sched_get_priority_max

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


"""
This function read the csv file 
file_path :  path from the csv file, should be string 
return a dataframe
"""
def load_datasets(file_path:str):
    df = pd.read_csv(file_path) #Load dataset
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle dataframe
    return df 

"""
This function display the dataframe shape, number element in the target and missing value
df :  dataframe 
target : target name, should be string
"""
def explore_datasets(df, target):
    print(df.shape) # display shape dataframe
    print(df[target].value_counts()) # count number element in the target 
    print(df.isnull().sum()) # check missing data

"""
This function display correlation and compute the weakest correlation
df :  dataframe 
"""
def display_correlation(df, target) :
    correlations = df[df.columns].corr(method='pearson')
    sns.heatmap(correlations, cmap="YlGnBu", annot = True)
    #plt.show()
    print('Absolute overall correlations')
    print('-' * 30)
    correlations_abs_sum = correlations[correlations.columns].abs().sum()
    print(correlations_abs_sum, '\n')
    print('Weakest correlations')
    print('-' * 30)
    print(correlations_abs_sum.nsmallest(3))
    print(df) 
