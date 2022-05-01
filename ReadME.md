

Overview of the current directory:

	household_power_consumption.txt
		Dataset that contains a household's recorded power consumption (recorded every minute of every day)
		Downloaded from : https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption


	power_consumption.csv
		The adapted csv for household_power_consumption.txt
			Missing values have been filled in with the previous record's appropriate value

		This CSV file was produced by the file prepare_dataset.py

	
	daily_power_consumption.py
		Instead of having the data be recorded by every minute, the data would be grouped into daily records

		This CSV file was produced by the file resample_csv.py

	prepare_dataset.py
		Converts the household_power_consumption.txt into a csv file, where missing values have been populated appropriately

		Output of this file is power_consumption.csv file

	resample_csv.py
		Reads in  the power_consumption.csv file
		Groups values in the dataset by day
		Writes these grouped records to a new csv (daily_power_consumption.csv)

	split_dataset.py
		Splits the daily dataset into weekly groups of power consumption levels
		The training dataset will contain data from the first three years in daily_power_consumption.csv
		The testing dataset will contain data from the final year in daily_power_consumption.csv

	convert_time_series_data.py
		Before the LSTM model can be trained, the time series forecasting problem must be
		reframed as a supervised learning problem.

		This is completed by converting the training dataset sequence to pairs of input and output sequences

		Output of this file is a return array of the input and output sequences


	plot_prediction_results.py
		Plots the RMSE value for different configurations of the LSTM model in the format of a grouped bar chart

	model.py
		Builds the Encoder-Decoder lstm model and saves it to the current working directory



	forecast_power_consumption.py
		Makes predictions on the next week's daily power consumption, outputs the actual consumption vs the predicted consumption
		as well as the MSE and RMSE for all predictions.

	300_epochs_4_batch_og_neurons
		The keras model



Please only run forecast_power_consumption.py if you want to see the predictions, but please visit the other mentioned python programs
in order to get more details on the process of completing this project, they have been well documented by comments and docstrings

All dependencies to run each program can be installed using the virtual environment MyVirtualEnv
