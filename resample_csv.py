
from pandas import read_csv
from pandas import DataFrame
from pandas.core.resample import DatetimeIndexResampler
#pandas.core.resample.DatetimeIndexResampler

def get_daily_groups(dataset: DataFrame) -> DatetimeIndexResampler:
    '''
    Group the dataset by each day rather than per minute 
    '''
    return dataset.resample("D")


def get_daily_dataset(resampled_dataset: DatetimeIndexResampler) -> DataFrame:
    '''
    Using the grouped by day dataset, sum all values per day and return a dataframe with each value corresponding to unique days
    '''
    return resampled_dataset.sum()



if __name__ == "__main__":
    #Read in csv
    power_consumption_dataset = read_csv("power_consumption.csv", header=0, infer_datetime_format=True, parse_dates=["datetime"], index_col=["datetime"])
    
    #Group values in the dataset by day
    dataset_grouped_per_day = get_daily_groups(power_consumption_dataset)

    #Get a new dataset with all values corresponding to individual days
    dataset_daily_groups_summed = get_daily_dataset(dataset_grouped_per_day)


    #Write the daily dataset to a new csv file
    dataset_daily_groups_summed.to_csv("daily_power_consumption.csv")
