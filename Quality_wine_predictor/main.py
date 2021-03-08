from datasets import *
from preprocessing import *
from models import *

if __name__ == "__main__":
    
    # load datasets
    df = load_datasets('./Wine-dataset/wine.csv')

    # display information about dataset
    explore_datasets(df, "quality")
    
    # display correlation and plot the graph
    display_correlation(df, "quality")

    # resample dataframe
    df = resampling(df, "quality")

    # split into test and train set 
    X_train, X_test, y_train, y_test = split_train_test(df, "quality", 0.8)

    # standardize X_train, X_test
    X_train, X_test = standard_scaler(X_train, X_test)

    # drop columns with a low correlation
    df = drop_low_correlation(df)

    # create and train the model
    model = wine_quality_predictor(X_train, y_train)

    # evaluate the keras model
    evaluate_accuracy(model, X_train, X_test, y_train, y_test)