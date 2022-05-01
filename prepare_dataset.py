'''

'''

from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric
from pandas import DataFrame

def get_dataset(text_file: str) -> DataFrame:
    '''
    Load data from household_power_consumption.txt file
    Separate on semi colon delimiter (sep=";")
    Use first row as the header (header=0)
    Parse columns 0 and 1 as a single date column (parse_dates={"datetime":[0,1]})
    Set datetime as the row label (index_col=["datetime"])
    Override default param of low_memory from True to false (low_memory=False)
    infer_datetime_format wil infer the format of datetime strings in columns and switch to a faster method of parsing them

    Return the pandas DataFrame
    '''
    dataset = read_csv(text_file, header=0, sep=";", parse_dates={"datetime":[0,1]}, index_col=["datetime"], low_memory=False, infer_datetime_format=True)
    return dataset




def replace_missing_values(dataset: DataFrame) -> DataFrame:
    '''
    Replace all missing values "?" with "nan"
    '''
    dataset.replace("?", nan, inplace=True)
    return dataset




def fill_missing_vals_in_dataframe(dataset_values: DataFrame) -> None:
    '''
    Loop through every row and column int he dataset's values
    Check if any value is nan
    If so, fill this missing value with yesterday's value for the same column

    No return type as we are passing by reference
    '''
    one_day  = 60 * 24

    print(f"One day: {str(one_day)}")
    
    #Loop through every row in the dataset
    for r in range(dataset_values.shape[0]):
        #Loop through every column in the dataset
        for c in range(dataset_values.shape[1]):
            #Check if nan exists within the current value being considered
            if isnan(dataset_values[r,c]):
                #Set nan value equal to yesterday's equivalent value
                dataset_values[r,c] = dataset_values[r-one_day, c]




def count_missing_values(text_file: str) -> None:
    '''
    Void function

    Open the dataset text file
    Loop through every line
    Count all lines with missing values
    Count all missing values
    Print total missing values and total lines with missing values
    
    '''
    lines_with_missing_values = 0
    missing_value_total = 0
    with open(text_file, "r") as f:
        for line in f:
            line_split = line.split(";")
            if "?" in line_split:
                lines_with_missing_values += 1
                missing_value_total += line_split.count("?")
    
    print(f"Number of lines with missing values: {str(lines_with_missing_values)}")
    
    print(f"\nTotal number of missing values: {str(missing_value_total)}")






if __name__ == "__main__":

    #Dataset downloaded from https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
    text_file = "household_power_consumption.txt"
    
    #count_missing_values(text_file)
    
    dataset = get_dataset(text_file)
    
    dataset_marked_missing_values = replace_missing_values(dataset)
    
    numeric_dataframe = dataset_marked_missing_values.astype("float32")

    fill_missing_vals_in_dataframe(numeric_dataframe.values)


    numeric_dataframe.to_csv("power_consumption.csv")



