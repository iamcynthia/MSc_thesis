import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.signal import butter,filtfilt,freqz

## butterworth low pass filter requirements
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq  # passband
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
# Filter requirements.
T = 100.0         # Sample Period
fs = 200.0       # sample rate, Hz
cutoff = 5      # desired cutoff frequency of the filter, Hz 
nyq = 0.5 * fs  # Nyquist Frequency
order = 4       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples   


################### 1. Segmentation & Label Given ###################
# Construct the file path
path = 'Annotated_Data/'
dirs = sorted(os.listdir(path))  ## Subject: SA01, SA02....
for index, item in enumerate(dirs):    ## pop the unuse file ".DS_Store"
    if item == '.DS_Store':
        dirs.pop(index)       
DATA_DIR_Ann = []  ## all file path
for names in dirs:
    info = os.path.join(path, names)
    for index, names in enumerate(os.listdir(info)):
        domain = os.path.join(info, names)
        DATA_DIR_Ann.append(domain)
DATA_DIR_Ann = sorted(DATA_DIR_Ann)

# Dataframe of the information of SisFall Dataset
ADLs = pd.read_csv('Activities.csv')

## create 2 dataframes: chunk dataset and feature_set dataset
Rough = pd.DataFrame()
chunk_R = pd.DataFrame()
for i in range(len(ADLs)):
    for index, item in enumerate(DATA_DIR_Ann): 
        if ADLs.iloc[i,0] in item:
            title = item.split("/")  ## to set plot title
            temp = pd.read_csv(item, header = None, dtype = float)  ## read all subject files
            temp.columns = ['acc1_X', 'acc1_Y', 'acc1_Z', 'gyr_X', 'gyr_Y', 'gyr_Z', 'acc2_X', 'acc2_Y', 'acc2_Z']
            temp['time'] = np.round(np.linspace(0.0, float(ADLs.iloc[i,3]), len(temp)),4)   ## set a new column for time
            temp = temp[['time', 'acc1_X', 'acc1_Y', 'acc1_Z', 'gyr_X', 'gyr_Y', 'gyr_Z', 'acc2_X', 'acc2_Y', 'acc2_Z']]
            temp = temp.drop(['acc2_X', 'acc2_Y', 'acc2_Z'], axis = 1)                  
            temp['Class'] = ADLs.iloc[i,7]  ## give roughly label

            # convert the unit from bit to gravity
            for j in range(len(temp)):
                temp.acc1_X[j] = ((2*32)/(2**13))*temp.acc1_X[j]
                temp.acc1_Y[j] = ((2*32)/(2**13))*temp.acc1_Y[j]
                temp.acc1_Z[j] = ((2*32)/(2**13))*temp.acc1_Z[j]
                temp.gyr_X[j] = ((2*4000)/(2**16))*temp.gyr_X[j]
                temp.gyr_Y[j] = ((2*4000)/(2**16))*temp.gyr_Y[j]
                temp.gyr_Z[j] = ((2*4000)/(2**16))*temp.gyr_Z[j]    

            ## implement butterworth lowpass filter and get filtered data
            filtered_X = butter_lowpass_filter(np.array(temp.acc1_X), cutoff, fs, order)  
            filtered_Y = butter_lowpass_filter(np.array(temp.acc1_Y), cutoff, fs, order)  
            filtered_Z = butter_lowpass_filter(np.array(temp.acc1_Z), cutoff, fs, order)  
            filtered_gyr_X = butter_lowpass_filter(np.array(temp.gyr_X), cutoff, fs, order) 
            filtered_gyr_Y = butter_lowpass_filter(np.array(temp.gyr_Y), cutoff, fs, order) 
            filtered_gyr_Z = butter_lowpass_filter(np.array(temp.gyr_Z), cutoff, fs, order) 

            ## add filtered data into dataframe temp and drop original columns
            temp['filtered_X'], temp['filtered_Y'], temp['filtered_Z'] = [filtered_X, filtered_Y, filtered_Z]
            temp['filtered_gyr_X'], temp['filtered_gyr_Y'], temp['filtered_gyr_Z'] = [filtered_gyr_X, filtered_gyr_Y, filtered_gyr_Z]
            temp = temp.drop(['acc1_X', 'acc1_Y', 'acc1_Z', 'gyr_X', 'gyr_Y', 'gyr_Z'], axis = 1)  
            chunk_R = pd.concat([chunk_R, temp])   ## store chunked data into dataframe chunk_R
            
            ## create dataframe of feature sets for each chunk               
            for s in range(0,len(temp),1000):
                df = pd.DataFrame(temp.iloc[s:s+1000,:])
                feat = np.array(df.describe().iloc[:,2:5].drop(['count','25%','50%', '75%'])).reshape(-1)
                label = df.describe().iloc[1,1].reshape(-1)
                stat = np.append(feat, label)
                Rough = Rough.append(pd.Series(stat), ignore_index=True)

Rough.columns = ['meanX', 'meanY', 'meanZ', 'stdX', 'stdY', 'stdZ' ,'minX', 'minY', 'minZ', 'maxX', 'maxY', 'maxZ', 'Class']  ## rename columns 

Rough.to_csv('dataset.csv', index = False)
chunk_R.to_csv('chunk_R.csv', index = False)




