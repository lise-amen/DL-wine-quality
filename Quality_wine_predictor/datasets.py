from os import sched_get_priority_max

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_datasets(file_path):
    #Load dataset
    df = pd.read_csv(file_path)
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle dataframe
    return df 

def explore_datasets(df):
    print(df.shape)
    print(df["quality"].value_counts())
    # check missing data
    df.isnull().sum()

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
        
