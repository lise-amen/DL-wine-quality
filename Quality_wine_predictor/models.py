import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# function to create a multi layer perceptron
def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(30, input_dim=dim, activation="relu"))
    # evaluate the keras model
    model.add(Dense(30, activation="relu"))
    model.add(Dense(30, activation="sigmoid"))
    model.add(Dense(7, activation="sigmoid"))
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
        print("Linear model")
    # return our model
    print('hello')
    return model