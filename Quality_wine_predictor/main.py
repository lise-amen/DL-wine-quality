from datasets import *
from preprocessing import *
from models import *

if __name__ == "__main__":
    
    df = load_datasets('./Wine-dataset/wine.csv')

    display_correlation(df, "quality")

    df = resampling(df, "quality")

    X_train, X_test, y_train, y_test = split_train_test(df, "quality")

    X_train, X_test = standard_scaler(X_train, X_test)

    df = df.drop(columns=['pH'])
    df = df.drop(columns=['residual sugar'])
    df = df.drop(columns=['sulphates'])

    model = create_mlp(12)

    # Build the model
    model.build()

    model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    #Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=200, epochs=250) 

    # evaluate the keras model
    _  , accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))

    # evaluate the keras model
    _, accuracy = model.evaluate(X_train, y_train)
    print('Accuracy: %.2f' % (accuracy*100))