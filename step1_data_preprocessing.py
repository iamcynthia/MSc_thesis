import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.signal import butter,filtfilt,freqz
import plotly.graph_objects as go
from sklearn.utils import resample
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks


################### 1. Convert txt files to csv files ###################
path = 'SisFall_dataset/'
dirs = os.listdir(path)
for index, item in enumerate(dirs):   ## pop the unuse file ".DS_Store"
    if item == '.DS_Store':
        dirs.pop(index)

## create a list of files
DATA_DIR = []        
for names in dirs:
    info = os.path.join(path, names)
    for index, names in enumerate(os.listdir(info)):
        domain = os.path.join(info, names)
        DATA_DIR.append(domain)

## create files
os.mkdir("Annotated_Data")
for k in dirs:
    os.mkdir("Annotated_Data/"+k)

## convert txt file to csv file for each subject
for names in DATA_DIR:
    if os.path.splitext(names)[1] == '.txt':
        in_file = open(names, 'r')
        stripped = (line.strip() for line in in_file)
        stripped = (line.strip(';') for line in stripped)
        lines = (line.split(",") for line in stripped if line)
        for item in dirs:
            if item in names:
                with open('Annotated_Data/' + os.path.splitext(names)[0].lstrip("SisFall_dataset") + ".csv", "w") as out_file:
                    writer = csv.writer(out_file)
                    writer.writerows(lines)


################### 2. Plots for all subjects and activities ###################
path = 'Annotated_Data/'
dirs = sorted(os.listdir(path))
for index, item in enumerate(dirs):    ## pop the unuse file ".DS_Store"
    if item == '.DS_Store':
        dirs.pop(index)
        
DATA_DIR_Ann = []
for names in dirs:
    info = os.path.join(path, names)
    for index, names in enumerate(os.listdir(info)):
        domain = os.path.join(info, names)
        DATA_DIR_Ann.append(domain)

fig , ax = plt.subplots()
fig.set_size_inches(20,40)
plt.subplots_adjust(wspace =0.2, hspace =0.5)

count = 0
for index, item in enumerate(sorted(DATA_DIR_Ann)):    
    if "D04" in item and "R01" in item:
        title = item.split("/")[2].split("_")
        activities = pd.read_csv(item, header = None, dtype = float)
        activities.columns = ['acc1_X', 'acc1_Y', 'acc1_Z', 'gyr_X', 'gyr_Y', 'gyr_Z', 'acc2_X', 'acc2_Y', 'acc2_Z']
        activities['time'] = np.round(np.linspace(0.0, 100.0, len(activities)),4)
        count +=1
        # select the data within 5 secs
        mask1 = activities['time'] <= 15.0000
        mask2 = activities['time'] >= 12.0000
        temp = activities[(mask1 & mask2)]
        temp.index = range(len(temp.index))

        for i in range(len(temp)):
            temp.acc1_X[i] = ((2*16)/(2**13))*temp.acc1_X[i]
            temp.acc1_Y[i] = ((2*16)/(2**13))*temp.acc1_Y[i]
            temp.acc1_Z[i] = ((2*16)/(2**13))*temp.acc1_Z[i]
#             temp.gyr_X[i] = ((2*2000)/(2**16))*temp.gyr_X[i]
#             temp.gyr_Y[i] = ((2*2000)/(2**16))*temp.gyr_Y[i]
#             temp.gyr_Z[i] = ((2*2000)/(2**16))*temp.gyr_Z[i]
#             temp.acc2_X[i] = ((2*8)/(2**14))*temp.acc2_X[i]
#             temp.acc2_Y[i] = ((2*8)/(2**14))*temp.acc2_Y[i]
#             temp.acc2_Z[i] = ((2*8)/(2**14))*temp.acc2_Z[i]
               
        ## plot it out
        plt.subplot(13, 3, count)
        temp.acc1_X.plot()
        temp.acc1_Y.plot()
        temp.acc1_Z.plot()
#         temp.gyr_X.plot()
#         temp.gyr_Y.plot()
#         temp.gyr_Z.plot()
        plt.title("%s" %(title[1]))
        plt.xlabel('samples')
        plt.ylabel('Acceleration [g]')
        # plt.legend()
        # plt.ylabel('Angular velocity [°/s]')
# plt.savefig("D04")


################### 3. Data Pre-processing -- Butterworth low pass filter ###################
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

## pakage: plotly
path = 'Annotated_Data/'
dirs = sorted(os.listdir(path))
for index, item in enumerate(dirs):    ## pop the unuse file ".DS_Store"
    if item == '.DS_Store':
        dirs.pop(index)
        
DATA_DIR_Ann = []
for names in dirs:
    info = os.path.join(path, names)
    for index, names in enumerate(os.listdir(info)):
        domain = os.path.join(info, names)
        DATA_DIR_Ann.append(domain)

# Filter the data, and plot both the original and filtered signals.
fig = go.Figure()

activities = pd.read_csv("Annotated_Data/SE06/F15_SE06_R05.csv", header = None, dtype = float)
activities.columns = ['acc1_X', 'acc1_Y', 'acc1_Z', 'gyr_X', 'gyr_Y', 'gyr_Z', 'acc2_X', 'acc2_Y', 'acc2_Z']
activities['time'] = np.round(np.linspace(0.0, 15.0, len(activities)),4)
for i in range(len(activities)):
    activities.acc1_X[i] = ((2*32)/(2**13))*activities.acc1_X[i]
    activities.acc1_Y[i] = ((2*32)/(2**13))*activities.acc1_Y[i]
    activities.acc1_Z[i] = ((2*32)/(2**13))*activities.acc1_Z[i]

Arr = np.array(activities.acc1_Y)
Arr1 = np.array(activities.acc1_X)
Arr2 = np.array(activities.acc1_Z)
y = butter_lowpass_filter(Arr, cutoff, fs, order)
x = butter_lowpass_filter(Arr1, cutoff, fs, order)
z = butter_lowpass_filter(Arr2, cutoff, fs, order)

# fig.add_trace(go.Scatter(y = Arr, line =  dict(shape =  'spline' ), name = 'raw signal Y'))
fig.add_trace(go.Scatter(y = Arr1, line =  dict(shape =  'spline' ), name = 'raw signal X'))
fig.add_trace(go.Scatter(y = Arr2, line =  dict(shape =  'spline' ), name = 'raw signal Z'))
fig.add_trace(go.Scatter(y = y, line =  dict(shape =  'spline' ), name = 'filtered signal Y'))
fig.add_trace(go.Scatter(y = x, line =  dict(shape =  'spline' ), name = 'filtered signal X'))
fig.add_trace(go.Scatter(y = z, line =  dict(shape =  'spline' ), name = 'filtered signal Z'))

fig.update_layout(
    title="Butterworth Lowpass Filter - Walking upstairs and downstairs quickly - Y",
    xaxis_title="Samples",
    yaxis_title="Acceleration [g]",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="#7f7f7f"
    )
)
fig.show()


################### 3. Data Pre-processing -- Moving Average Smoothing ###################
activities = pd.read_csv("Annotated_Data/SA06/D01_SA06_R01.csv", header = None, dtype = float)
activities.columns = ['acc1_X', 'acc1_Y', 'acc1_Z', 'gyr_X', 'gyr_Y', 'gyr_Z', 'acc2_X', 'acc2_Y', 'acc2_Z']
activities['time'] = np.round(np.linspace(0.0, 100.0, len(activities)),4)
activities = activities[['time','acc1_X', 'acc1_Y', 'acc1_Z', 'gyr_X', 'gyr_Y', 'gyr_Z', 'acc2_X', 'acc2_Y', 'acc2_Z']]
for i in range(len(activities)):
#     activities.acc1_X[i] = activities.acc1_X[i] * ((2*32)/(2**13))
    activities.acc1_Y[i] = activities.acc1_Y[i] * ((2*32)/(2**13))
    activities.acc1_Z[i] = activities.acc1_Z[i] * ((2*32)/(2**13))

# Tail-rolling average transform
data = activities.loc[:,('time','acc1_X', 'acc1_Y', 'acc1_Z')]
# data['filtered_X'] = data.acc1_X.rolling(20).mean()
data['filtered_Y'] = data.acc1_Y.rolling(15).mean()
data['filtered_Z'] = data.acc1_Z.rolling(15).mean()

# plot original and transformed dataset
fig = go.Figure()
fig.add_trace(go.Scatter(x = data.time, y = activities.acc1_Y, line =  dict(shape =  'spline' ), name = 'raw signal - Y'))
fig.add_trace(go.Scatter(x = data.time, y = data.filtered_Y, line =  dict(shape =  'spline' ), name = 'filtered signal - Y'))
fig.add_trace(go.Scatter(x = data.time, y = activities.acc1_Z, line =  dict(shape =  'spline' ), name = 'raw signal - Z'))
fig.add_trace(go.Scatter(x = data.time, y = data.filtered_Z, line =  dict(shape =  'spline' ), name = 'filtered signal - Z'))

fig.update_layout(
    title="Moving Average Filter - Walking upstairs and downstairs quickly - Y",
    xaxis_title="Time [s]",
    yaxis_title="Acceleration [g]",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="#7f7f7f"
    )
)
fig.show()


################### 3. Data Pre-processing -- Downsampling ###################
data = pd.read_csv("Annotated_Data/SA06/D01_SA06_R01.csv", header = None, dtype = float) ## use one as an example
data.columns = ['acc1_X', 'acc1_Y', 'acc1_Z', 'gyr_X', 'gyr_Y', 'gyr_Z', 'acc2_X', 'acc2_Y', 'acc2_Z']
data['time'] = np.round(np.linspace(0.0, 100.0, len(data)),4)  ## set a new column for time
for i in range(len(data)):
    data.acc1_X[i] = data.acc1_X[i] * ((2*32)/(2**13))
    data.acc1_Y[i] = data.acc1_Y[i] * ((2*32)/(2**13))
    data.acc1_Z[i] = data.acc1_Z[i] * ((2*32)/(2**13))

## pakage: plotly
# Filter the data, and plot both the original and filtered signals.
fig = go.Figure()
ArrY = np.array(data.acc1_Y)
ArrZ = np.array(data.acc1_Z)
data_downsampled = pd.DataFrame()  ## empty dataframe for down-sampled data
for k in range(0,len(data), 200):  ## sample for every 200 seconds
    df = data[k : k+200]
    df1 = resample(df, replace =False, n_samples = 20, random_state = 0)
    df1 = df1.sort_values(by = ['time'])
    data_downsampled = pd.concat([data_downsampled, df1])
    data_downsampled.index = range(len(data_downsampled))
fig.add_trace(go.Scatter(x = data.time, y = ArrY, line =  dict(shape =  'spline' ), name = 'raw signal - Y'))
fig.add_trace(go.Scatter(x = data_downsampled.time, y = data_downsampled.acc1_Y, line =  dict(shape =  'spline' ), name = 'filtered signal - Y'))
fig.add_trace(go.Scatter(x = data.time, y = ArrZ, line =  dict(shape =  'spline' ), name = 'raw signal - Z'))
fig.add_trace(go.Scatter(x = data_downsampled.time, y = data_downsampled.acc1_Z, line =  dict(shape =  'spline' ), name = 'filtered signal - Z'))

fig.update_layout(
    title="Downsampling",
    xaxis_title="Time [s]",
    yaxis_title="Acceleration [g]",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="#7f7f7f"
    )
)
fig.show()


################### 4. Peak Detection ################### 
x = butter_lowpass_filter(np.array(activities.acc1_Y), cutoff, fs, order)  ## can be any axis which you want to find peaks
peaks, _= find_peaks(x)
allPeaks_mean = x[peaks].mean()
peaks, _ = find_peaks(x, height = allPeaks_mean)
std = np.std(x[peaks])
fig = go.Figure()
fig.add_trace(go.Scatter(y = x, line =  dict(shape =  'spline' ), name = 'filtered signal'))
fig.add_trace(go.Scatter(x = peaks, y = x[peaks], mode = 'markers', name = 'peaks'))
fig.add_shape(type = 'line', x0 = 0, y0 = allPeaks_mean + 1/2 * std, x1 = max(peaks), y1 = allPeaks_mean + 1/2 *std, line=dict(color='green'))
# fig.add_shape(type = 'line', x0 = 0, y0 = allPeaks_mean + 1/6 *std, x1 = max(peaks), y1 = allPeaks_mean + 1/6 *std, line=dict(color='orange'))

fig.update_layout(
    title="Peak Detection",
    xaxis_title="Samples",
    yaxis_title="Acceleration [g]",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="#7f7f7f"
    )
)
fig.show()





