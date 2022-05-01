from pandas import read_csv
from numpy import array
from split_dataset import split_dataset
from convert_timeseries_data import convert_time_series_to_supervised
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import adam_v2

# train the model
def build_multi_input_encoder_decoder(train_x, train_y, verbose, epochs, batch_size):
    '''Function that will build and return a multiple input encoder-decoder LSTM
    The model consists of two sub-models
        The encoder - which reads and encodes the input sequence
        The decoder - to read the encoded input sequence
        Using this model, we will be able to forecast the power consumption
        for the next n amount of days'''
    
    '''train_y must be reshapedto have a three dimensional structure instead of the already 
    existing structure of [samples, features]
    train_y's new three dimensional shape is to match the single week prediction shape of [1,7,1]'''
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    #Initialize model
    model = Sequential()
    ''' Define a hidden LSTM Encoder layer with 200 units
            -The encoder model will read the input sequence and outputs a 200 element vector 
            (with one node per unit) that captures features from the input sequence
        The input shape will be defined by the numeric value passed by the train_x[1] and train_x[2] shapes'''
    model.add(LSTM(200, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    ''' Next, the representation of the input sequence (train_x) will be repeated for each time step in the 
        output sequence (train_y) by adding a RepeatVector layer to the model'''
    n_outputs = train_y.shape[1]
    model.add(RepeatVector(n_outputs))
    ''' Define the decoder layer with 200 units
        Decoder will output the entire sequence and not just the output at the end of the sequence as 
        seen when defining the encoder layer. This means each unit in the decoder layer will output a 
        value for each day to be forecasted which represents the basis for  what to predict for each day 
        in the output sequence  '''
    model.add(LSTM(200, activation='relu', return_sequences=True))
    ''' model.add(TimeDistributed(Dense(100, activation="relu"))) does the following:
        The TimeDistributed layer will interpret each time step in the output sequence (train_y) '''
    model.add(TimeDistributed(Dense(200, activation='relu')))
    #Output layer
    model.add(TimeDistributed(Dense(1, activation="linear")))
    '''MSE has been chosen as the loss function as it will be used to ensure that the trained model has no 
        outlier predictions with huge errors '''
    model.compile(loss='mse', optimizer='adam')
    #Train model
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model






if __name__ == "__main__":
    #Read in dataset
    dataset = read_csv("daily_power_consumption.csv", header=0, infer_datetime_format=True, parse_dates=["datetime"], index_col=["datetime"])

    #Split dataset into training and testing sets
    training_dataset, testing_dataset = split_dataset(dataset)

    train_x, train_y = convert_time_series_to_supervised(training_dataset, 14)

    model = build_multi_input_encoder_decoder(train_x, train_y, 1, 300 ,16)

    model.save("new_model")
