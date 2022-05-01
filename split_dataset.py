from pandas import read_csv
from numpy import split
from numpy import array

def split_dataset(dataset):
    dataset_values = dataset.values

    #Training set will consist of all records within first three years within dataset
    training_set = dataset_values[0:-329]
    

    #Testing data will consist of all records after the first three years within the dataset
    testing_set = dataset_values[-329:]
    

    #training_set_weekly will consist of the training_set's weekly data (159 weeks worth of data)
    training_set_weekly = array(split(training_set, len(training_set)/7))

    #testing_set weekly will consist of the testing_set's weekly data (47 weeks worth of data)
    testing_set_weekly = array(split(testing_set, len(testing_set)/7))

    return [training_set_weekly, testing_set_weekly]




if __name__ == "__main__":
    dataset = read_csv("daily_power_consumption.csv", header=0, infer_datetime_format=True, parse_dates=["datetime"], index_col=["datetime"])
    training_dataset, testing_dataset = split_dataset(dataset)

    print(f"Total number of records in the training dataset: {str(len(training_dataset))} (weeks)")
    print(f"Training dataset's shape: {str(training_dataset.shape)}")
    print(f"Total number of records in the testing dataset: {str(len(testing_dataset))} (weeks)")
    
