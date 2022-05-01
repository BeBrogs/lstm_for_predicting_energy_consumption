from numpy import array
from numpy import arange
import matplotlib.pyplot as plt
from numpy import mean
from sklearn.metrics import mean_squared_error as mse
from math import sqrt


#Labels
days = [" ", "Sat", "Sun", "Mon", "Tues", "Weds", "Thurs", "Fri"]

#Actual results of the next week's power consumption
actual = [1309, 2083, 1604, 2219, 1777, 1769, 1797]

def plot_predicted_vs_actual_32_batch_size():

    #Predictions with 50 epochs
    fifty_epochs = [1896, 1907, 1917, 1878, 1866, 1869, 1865]
    
    #Predictions with 300 epochs
    three_hundred_epochs = [1612, 1602, 1597, 1571, 1557, 1578, 1568]
    
    #Predictions with 150 epochs
    two_hundred_epochs = [1772, 1905, 1931, 2003, 2046, 2070, 2088]
    
    #Predicts with 500 epochs
    one_hundred_epochs = [1670, 1680, 1623, 1695, 1627, 1594, 1650]

    #plot_epoch_data(fifty_epochs, one_hundred_epochs, two_hundred_epochs, three_hundred_epochs)

    print("\t32 batch size, 200 neurons encoder decoder\t".center(80, "="))

    fifty_epoch_rmse = get_rmse(fifty_epochs)
    print(f"50 epoch rmse {str(fifty_epoch_rmse)}")


    one_hundred_rmse = get_rmse(one_hundred_epochs)
    print(f"100 epoch rmse {str(one_hundred_rmse)}")


    two_hundred_rmse = get_rmse(two_hundred_epochs)
    print(f"200 epoch rmse {str(two_hundred_rmse)}")


    three_hundred_rmse = get_rmse(three_hundred_epochs)
    print(f"300 epoch rmse {str(three_hundred_rmse)}")

    return [fifty_epoch_rmse, one_hundred_rmse, two_hundred_rmse, three_hundred_rmse]



def get_rmse(data):    
    return sqrt(mse(actual, data))



'''
def plot_rmse_32_batch_size():
    rmse_50_epochs = array([549, 442, 394, 394, 427, 401, 337])
    rmse_100_epochs = array([557,465, 455, 397, 459, 409, 362])
    rmse_200_epochs = array([506, 469, 499, 534, 522, 618, 588])
    rmse_300_epochs = array([552, 466, 455, 422, 482, 431, 371])

    avg_rmse_50_epochs = rmse_50_epochs.mean()
    avg_rmse_100_epochs = rmse_100_epochs.mean()
    avg_rmse_200_epochs = rmse_200_epochs.mean()
    avg_rmse_300_epochs = rmse_300_epochs.mean()

    print("".center(80, "="))
    print("Average RMSEs for batch size 32 (Original Neurons)")
    print_rmse_values_per_epoch(avg_rmse_50_epochs, avg_rmse_100_epochs, avg_rmse_200_epochs, avg_rmse_300_epochs)
'''

def plot_rmse_data():
    categories = ["50 epochs", "100 epochs", "200 epochs", "300 epochs"]

    batch_16_small_neurons = plot_predicted_vs_actual_16_batches_small_neurons()

    batch_32_og_neurons = plot_predicted_vs_actual_32_batch_size()

    batch_4_more_neurons = plot_predicted_vs_actual_4_batches_og_neurons()

    batch_4_og_neurons = plot_predicted_vs_actual_4_batches_more_neurons()

    X = arange(4)
    
    fig, ax = plt.subplots()
    #fig = plt.figure()
    #ax = fig.add_axes([0,0,1,1])
    group_zero = ax.bar(X - 0.2, batch_16_small_neurons, width=0.2, label="16 batch size, 50 neurons", edgecolor="black")
    group_one = ax.bar(X  , batch_32_og_neurons, width=0.2, label="32 Batch Size, 200 neurons", edgecolor="black")
    group_three = ax.bar(X + 0.2, batch_4_og_neurons, width=0.2, label="4 Batch Size, 400 Neurons", edgecolor="black")
    group_two = ax.bar(X + 0.4, batch_4_more_neurons, width=0.2, label="4 Batch Size, 200 Neurons", edgecolor="black")

    ax.set_ylabel("RMSE")
    ax.set_xlabel("TEST")
    ax.legend()

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    #ax.set_xticklabels(categories)
    fig.tight_layout()
    #plt.xticks(array(batch_4_og_neurons))
    plt.xticks(X, categories)
    plt.show()


def plot_epoch_data(fifty_epochs, one_hundred_epochs, two_hundred_epochs, three_hundred_epochs):
    x = arange(7)
    width=0.15
    fig, ax = plt.subplots()
    consumption_actual = ax.bar(x - 0.3, actual, width, label="Actual Consumption", color="black")
    consumption_50_epochs = ax.bar(x - 0.15, fifty_epochs, width, label="50 Epochs Prediction")
    consumption_100_epochs = ax.bar(x, one_hundred_epochs, width, label="100 Epochs Prediction") 
    consumption_200_epochs = ax.bar(x + 0.15, two_hundred_epochs, width, label="200 Epochs Prediction")
    consumption_300_epochs = ax.bar(x + 0.3, three_hundred_epochs, width, label = "300 Epochs Prediction")

    ax.set_ylabel("Consumption")
    ax.set_xlabel("Days")
    ax.legend()


    ax.set_xticklabels(days)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    fig.tight_layout()
    plt.show()




def plot_predicted_vs_actual_4_batches_og_neurons():
    #Predictions with 50 epochs
    fifty_epochs = [2017, 1997, 1953, 1883, 1909, 2100, 2100]

    #Predictions with 100 epochs
    one_hundred_epochs = [1428, 1507, 1526, 1514, 1518, 1527, 1535]

    #Prediction with 200 epochs
    two_hundred_epochs = [1661, 2069, 1970, 1719, 1997, 2085, 1812]

    #Prediction with 300 epochs
    #three_hundred_epochs = [1757, 1767, 1688, 1707, 1711, 1710, 1710]
    three_hundred_epochs = [1551, 1717, 1664, 1640, 1639, 1639, 1639]


    #plot_epoch_data(fifty_epochs, one_hundred_epochs, two_hundred_epochs, three_hundred_epochs)
    print("\n\n")
    print("\t4 batch size, 200 neurons encoder-decoder\t".center(80, "="))


    fifty_epoch_rmse = get_rmse(fifty_epochs)
    print(f"50 epoch rmse {str(fifty_epoch_rmse)}")


    one_hundred_epoch_rmse = get_rmse(one_hundred_epochs)
    print(f"100 epoch rmse {str(one_hundred_epoch_rmse)}")


    two_hundred_epoch_rmse = get_rmse(two_hundred_epochs)
    print(f"200 epoch rmse {str(two_hundred_epoch_rmse)}")


    three_hundred_epoch_rmse = get_rmse(three_hundred_epochs)
    print(f"300 epoch rmse {str(three_hundred_epoch_rmse)}")

    return [fifty_epoch_rmse, one_hundred_epoch_rmse, two_hundred_epoch_rmse, three_hundred_epoch_rmse]

'''
def plot_rmse_4_batches_more_neurons():
    rmse_50_epochs = array([571, 616, 629, 565, 589, 504, 414])
    rmse_100_epochs = array([667, 546, 454, 448, 542, 470, 445])
    rmse_200_epochs = array([344, 411, 356, 320, 355, 367, 267])
    rmse_300_epochs = array([415, 413, 394, 404, 443, 396, 332])

    avg_rmse_50_epochs = rmse_50_epochs.mean()
    avg_rmse_100_epochs = rmse_100_epochs.mean()
    avg_rmse_200_epochs = rmse_200_epochs.mean()
    avg_rmse_300_epochs = rmse_300_epochs.mean()
   
    print("".center(80, "="))
    print("Average RSMEs for batch size 4 (More Neurons)")
   
    print_rmse_values_per_epoch(avg_rmse_50_epochs, avg_rmse_100_epochs, avg_rmse_200_epochs, avg_rmse_300_epochs)
'''


def print_rmse_values_per_epoch(avg_rmse_50_epochs, avg_rmse_100_epochs, avg_rmse_200_epochs, avg_rmse_300_epochs):
    print(f"Average RMSE for 50 epochs: {str(round(avg_rmse_50_epochs, 2))}")
    print(f"Average RMSE for 100 epochs: {str(round(avg_rmse_100_epochs, 2))}")
    print(f"Average RMSE for 200 epochs: {str(round(avg_rmse_200_epochs,2))}")
    print(f"Average RMSE for 300 epochs: {str(round(avg_rmse_300_epochs,2))}")
    print("\n")


def plot_predicted_vs_actual_4_batches_more_neurons():
    #Predictions with 50 epochs
    fifty_epochs = [1632, 1641, 1666, 1666, 1666, 1666, 1666]

    #Predictions with 100 epochs
    one_hundred_epochs = [1894, 1860, 1769, 1767, 1767, 1767, 1767]

    #Predictions with 200 epochs
    two_hundred_epochs = [1397, 1649, 1576, 1574, 1577, 1577, 1577]

    #Predictions with 300 epochs
    three_hundred_epochs = [1626, 1532, 1532, 1532, 1532, 1532, 1632]
    #three_hundred_epochs = [1551, 1717, 1664, 1640, 1639, 1639, 1639]
    #Plot data
    #plot_epoch_data(fifty_epochs, one_hundred_epochs, two_hundred_epochs, three_hundred_epochs)
   
    print("\t4 batch size, 400 neurons\t".center(80, "="))

    fifty_epoch_rmse = get_rmse(fifty_epochs)
    print(f"50 epoch rmse {str(fifty_epoch_rmse)}")

    one_hundred_epoch_rmse = get_rmse(one_hundred_epochs)
    print(f"100 epoch rmse {str(one_hundred_epoch_rmse)}")
    
    two_hundred_epoch_rmse = get_rmse(two_hundred_epochs)
    print(f"200 epoch rmse {str(two_hundred_epoch_rmse)}")

    three_hundred_epoch_rmse = get_rmse(three_hundred_epochs)
    print(f"300 epoch rmse {str(three_hundred_epoch_rmse)}")

    return [fifty_epoch_rmse, one_hundred_epoch_rmse, two_hundred_epoch_rmse, three_hundred_epoch_rmse] 


def plot_predicted_vs_actual_16_batches_small_neurons():
    fifty_epochs = [1602, 1560, 1564, 1612, 1580, 1575, 1647]

    one_hundred_epochs = [1640, 1408, 1526, 1453, 1397, 1346, 1273]

    two_hundred_epochs = [1899, 1927, 1924, 1924, 1925, 1925, 1925]

    three_hundred_epochs = [1901, 1898, 1866, 1854, 1851, 1849, 1849]

    
    fifty_epoch_rmse = get_rmse(fifty_epochs)
    
    one_hundred_epoch_rmse = get_rmse(one_hundred_epochs)

    two_hundred_epoch_rmse = get_rmse(two_hundred_epochs)
    
    three_hundred_epochs_rmse = get_rmse(three_hundred_epochs)

    return [fifty_epoch_rmse, one_hundred_epoch_rmse, two_hundred_epoch_rmse, three_hundred_epochs_rmse]






'''
def plot_rmse_4_batches():
    rmse_50_epochs = array([572, 499, 514, 496, 523, 510, 460])
    rmse_100_epochs = array([424, 410, 360, 322, 391, 369, 322])
    rmse_200_epochs = array([411, 400, 362, 326, 380, 363, 280])
    rmse_300_epochs = array([413, 416, 362, 335, 385, 365, 282])
    
    avg_rmse_50_epochs = rmse_50_epochs.mean()
    avg_rmse_100_epochs = rmse_100_epochs.mean()
    avg_rmse_200_epochs = rmse_200_epochs.mean()
    avg_rmse_300_epochs = rmse_300_epochs.mean()

    print("".center(80, "="))
    print("Average RMSE for batch size 4 (Original Size Neurons)")
    print_rmse_values_per_epoch(avg_rmse_50_epochs, avg_rmse_100_epochs, avg_rmse_200_epochs, avg_rmse_300_epochs)
'''



if __name__ == "__main__":
    plot_rmse_data() 


    #plot_predicted_vs_actual_32_batch_size()
    #plot_predicted_vs_actual_4_batches()
    #plot_predicted_vs_actual_4_batches_more_neurons()

    #plot_rmse_32_batch_size()
    #plot_rmse_4_batches()
    #plot_rmse_4_batches_more_neurons()
