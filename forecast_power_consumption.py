#Package imports
from pandas import read_csv
from numpy import array
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras_visualizer import visualizer
import graphviz
#from ann_visualizer.visualize import ann_viz
#Module imports
from split_dataset import split_dataset
from model import build_multi_input_encoder_decoder
from convert_timeseries_data import convert_time_series_to_supervised






def forecast_power_consumption(model, num_days_to_predict, training_dataset, testing_dataset) -> None:
    '''  '''    
    
    #List of all previous weekly data
    weekly_consumption_data = [i for i in training_dataset]

    #Array to hold the forecasted week's data
    forecasted_predictions = []
    

    #Loop over 
    for i in range(len(testing_dataset)):
        #
        predicted_consumption = predict_power_consumption(model, weekly_consumption_data, num_days_to_predict)
        
        #Append forecast to list
        forecasted_predictions.append(predicted_consumption) 

        #Add real prediction to weekly_consumption_data, will be used in predict_power_consumption to predict next week's forecast
        this_weeks_consumption = testing_dataset[i, :]
        weekly_consumption_data.append(this_weeks_consumption)


    predicted_consumption_np = array(forecasted_predictions)
    print_forecast(testing_dataset[:,:,0], predicted_consumption_np)


def predict_power_consumption(model, weekly_consumption_data, num_days_to_predict) -> array:
    '''  '''
    
    #Convert weekly_consumption_data from list to numpy array
    weekly_data_np = array(weekly_consumption_data)

    #Flatten data to parse previous week in weekly_data_np
    flattened_weekly_data = weekly_data_np.reshape((weekly_data_np.shape[0] * weekly_data_np.shape[1], weekly_data_np.shape[2])) 

    #Get previous week of data
    previous_week = flattened_weekly_data[-num_days_to_predict:, :]
    
    #Reshape previous week in order to make prediction
    previous_week_reshaped = previous_week.reshape((1,previous_week.shape[0], previous_week.shape[1]))

    #Forecast the next week of power consumption
    predicted_values = model.predict(previous_week_reshaped, verbose=0)
    
    #Return vector forecast
    return predicted_values[0]



def validate_total_weeks(total_weeks) -> None:
    try:
        int(total_weeks)
    except:
        raise ValueError("Error, total weeks should be a whole number  (You entered {str(total_weeks))")





def print_forecast(actual, predicted) -> None:
    forecasts = []
    rmse_arr = []

    #Ask user for total number of weeks to forecast
    #total_weeks = input("How many weeks of power consumption would you like to predict?")


    #Validate user input
    #validate_total_weeks(total_weeks)


    day_labels = ["Sat", "Sun", "Mon", "Tues", "Weds", "Thurs", "Fri"]
    for week in range(2):
        this_weeks_data = []
        #print(f"Week {str(week)})")
        for i in range(actual.shape[1]):
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            this_weeks_data.append({"Day": day_labels[i], "actual consumption": round(actual[:, i][week], 2), "predicted consumption": round(predicted[:, i][week][0], 2)})#, "MSE": mse}])

            rmse = sqrt(mse)
            rmse_arr.append(rmse)

        forecasts.append(this_weeks_data)



    keys = ["Day", "Actual Consumption", "Predicted Consumption"]
    for enum, i in enumerate(forecasts):
        print(f"\nWeek {str(enum + 1)}")
        print("\t\t".join(keys))
        
        actual_consumption = []
        predicted_consumption = []
        for j in i:
            values = []
            for enum_k, k in enumerate(j.values()):
                if enum_k < 2: 
                    values.append(str(k))
                    if enum_k ==1:
                        actual_consumption.append(k)
                else:
                    values.append(f"\t\t{str(k)}")
                    predicted_consumption.append(k)


            print("\t\t".join(values))
        #print(actual_consumption)
        #print(predicted_consumption)
        print(f"\nWeek {str(enum+1)} MSE: {str(round(mean_squared_error(actual_consumption, predicted_consumption), 2))} killowats")
        print(f"Week {str(enum+1)} RMSE: {str(round(sqrt(mean_squared_error(actual_consumption, predicted_consumption)),2))} killowats")



    print("\n")

if __name__=="__main__":
    
    #Read csv into a pandas dataframe
    dataset = read_csv("daily_power_consumption.csv", header=0, infer_datetime_format=True, parse_dates=["datetime"], index_col=["datetime"])

    #Split dataset into train and test datasets
    training_dataset, testing_dataset = split_dataset(dataset)

    #Determine number of days to forecast
    num_of_days_to_forecast = 14


    #Split dataset into inputs and outputs
    input_seq, output_seq = convert_time_series_to_supervised(training_dataset, num_of_days_to_forecast)

    #Load in trained model
    
    #Model for coursework
    #model = load_model("200_epochs_4_batch_more_neurons")
    #model = load_model("300_epochs_4_batch_more_neurons")
    
    #THIS ONE
    model = load_model("300_epochs_4_batch_og_neurons")
    #model = load_model("50_epochs_16_batch_og_neurons")

    #model = load_model("50_epochs_16_batch_small_neurons")
    #model = load_model("200_epochs_16_batch_small_neurons")
    #model = load_model()
    #model = load_model("50_epochs_16_batch_og_neurons")

    #Forecast power consumption for the next <number_of_days_to_forecast> days
    forecast_power_consumption(model, num_of_days_to_forecast, training_dataset, testing_dataset)
    #ann_viz(model, title="Model")
    plot_model(model, to_file="model_plit.png", show_shapes=True)
