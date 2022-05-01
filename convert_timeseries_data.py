from pandas import read_csv
from numpy import array
from split_dataset import split_dataset



def convert_time_series_to_supervised(training_dataset, total_num_days, n_out = 7):
    '''
    Before the LSTM model can be trained, the time series forecasting problem must be
    reframed as a supervised learning problem.
    
    This is completed by converting the sequence to pairs of input and output sequences
    '''

    #Flatten the data
    training_data_reshaped = training_dataset.reshape((training_dataset.shape[0]*training_dataset.shape[1], training_dataset.shape[2]))

    #Lists to hold input and output sequences
    train_x = []
    train_y = []


    '''
    Iterate over the training_data_reshaped data in order to increase the training data set
        i.e. instead of just considering the standard weeks in our dataset, we are able to loop over an array of values from 
        the current week following the current day, eg:

            Current day == 3, Current week following == [day3, day4, day5, day6, day7, day8, day9]
            The total number of days in each sequence depends on param: total_num_days, in the above example total_num_days is equal to 7


    This will increase the accuracy of the LSTM forecasting as instead of training the Model with the gathered 159 weeks of data,
    the total number of training records will be equal to (len(training_dataset) * total_num_days) 
        i.e.: When the len(training_dataset) = 159 and total_num_days = 7, then we have 1100 total number of records to train the dataset
    '''
    
    length_of_dataset = len(training_data_reshaped)

    in_start = 0
    #Iterate dataset
    for indx in range(length_of_dataset):
        
        #Define where the sequences should start and end
        train_x_range_end = in_start + total_num_days
        train_y_range_end = train_x_range_end + n_out

        #Validate that there is enough data to append to train_x and train_y
        if train_y_range_end <= len(training_data_reshaped):
            train_x_input = training_data_reshaped[indx: train_x_range_end, :]
            
            train_x.append(train_x_input)


            train_y_input = training_data_reshaped[train_x_range_end:train_y_range_end, 0]
            train_y.append(train_y_input)
        in_start +=1
    
    #Convert lists to numpy arrays
    train_x, train_y = array(train_x), array(train_y)
    

    return [train_x, train_y]

    
if __name__ == "__main__":
    dataset = read_csv("daily_power_consumption.csv", header=0, infer_datetime_format=True, parse_dates=["datetime"], index_col=["datetime"])

    training_dataset, testing_dataset = split_dataset(dataset)
    train_x, train_y = convert_time_series_to_supervised(training_dataset, 7)
    print(train_x.shape)
    print(train_y.shape)

