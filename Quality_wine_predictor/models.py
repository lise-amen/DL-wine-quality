import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

"""
This function create a multi layer perceptron
dim :  dimension from the target set  
regress : if True create a linear model, should be boolean values 
fraction : fraction from train dataset, should be float
return model
"""
def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(30, input_dim=dim, activation="relu"))
    # evaluate the keras model
    model.add(Dense(30, activation="relu"))
    model.add(Dense(30, activation="relu"))
    model.add(Dense(7, activation="sigmoid"))
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
        print("Linear model")
    # return our model
    print('hello')
    return model

"""
This function create a neural networks to predict wine quality
X_train : train set, should be dataframe
X_test : train set, should be a dataframe
return model
"""
def wine_quality_predictor(X_train, y_train):
    model = create_mlp(12)

    model.build() # Build the model

    model.summary()

    #Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=200, epochs=250) 

    return model 

"""
This function display the accuracy score for train and test set
X_train : train set, should be dataframe
X_test : train set, should be a dataframe
y_train : test set, should be a dataframe
y_test : test set, should be a dataframe
return model
"""
def evaluate_accuracy(model, X_train, X_test, y_train, y_test):
    _  , accuracy = model.evaluate(X_test, y_test)
    print('Accuracy test set : %.2f' % (accuracy*100))

    # evaluate the keras model
    _, accuracy = model.evaluate(X_train, y_train)
    print('Accuracy train set : %.2f' % (accuracy*100))