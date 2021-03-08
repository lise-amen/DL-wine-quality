from sklearn.utils import resample 
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

"""
This function split the dataframe into train and test train
df :  dataframe 
target : target name, should be string
fraction : fraction from train dataset, should be float
return  X_train, X_test, y_train, y_test
"""
def split_train_test(df, target, fraction) :
    # Creating a test/train split
    train_test_split_fraction = fraction
    split_index = int(df.shape[0] * train_test_split_fraction)
    df_train = df[:split_index]
    df_test = df[split_index:]

    # Selecting the features and the target
    X_train = df_train.drop(target, axis = 1).values
    X_test = df_test.drop(target, axis = 1).values
    target = pd.get_dummies(df[target]).values # One hot encode
    y_train = target[:split_index]
    y_test = target[split_index:]

    return X_train, X_test, y_train, y_test

"""
This function adding more sample from the minority class to resample dataframe
df :  dataframe 
target : target name, should be string
return  dataframe
"""
def resampling(df, target):
    df_majority = df[df[target] == 6]
    for i in range(3,10):
        majority_len = df[df[target] == 6].shape[0]
        if i != 6:
            df_minority = df[df[target] == i]
                 
            df_minority_upsampled = resample(df_minority,
                                            replace=True,
                                            n_samples=majority_len,
                                            random_state=123)
                 
            df_majority = df_majority.append(df_minority_upsampled)
    df = df_majority
    print(df.shape)
    print(df[target].value_counts())
    df = df.sample(frac=1).reset_index(drop=True)
    return df 

"""
This function transform your data with a distribution with a mean value 0 and standard deviation of 1
X_train : train set, should be dataframe
X_test : train set, should be a dataframe
return  X_train, X_test
"""
def standard_scaler(X_train, X_test) :
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)  
    return  X_train, X_test

"""
This function drop columns pH, residual sugar, sulphates
df : dataframe set, should be a dataframe
return  dataframe
"""
def drop_low_correlation(df):
    df = df.drop(columns=['pH'])
    df = df.drop(columns=['residual sugar'])
    df = df.drop(columns=['sulphates'])
    return df 