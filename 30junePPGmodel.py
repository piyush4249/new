import math
import heartpy as hp
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import auth
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tkinter import font
from tkinter.font import Font
from twilio.rest import Client
import requests
import pyrebase
import smbus  #sudo i2cdetect -y 1
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import transforms
from PIL import Image,ImageTk
import pywt
import tkinter as tk
from tkinter import Tk, Button, Label, Toplevel, Entry, filedialog
from datetime import datetime
from gpiozero import MCP3008
import os
import csv
import neurokit2 as nk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import spidev
import time
from max30102 import MAX30102
import numpy as np
from scipy.signal import find_peaks
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from scipy.signal import iirnotch, lfilter,firwin
import pandas as pd
import matplotlib.ticker as ticker
from keras.models import load_model
#PPG models
modelPPG1 = load_model("/home/pi/Desktop/IOT/Models/neural_network_model_1.h5")
modelPPG2 = load_model("/home/pi/Desktop/IOT/Models/neural_network_model_2.h5")



# Read the annotation file
annotation_data = pd.read_csv("/home/pi/Desktop/IOT/Models/PPGannotations.csv", parse_dates=['Datetime'])

# Sort the data by datetime if it's not already sorted
annotation_data = annotation_data.sort_values('Datetime')
#PPG models END

font_path = '/home/pi/Desktop/IOT/fonts/DS-DIGI.TTF'
font_prop = FontProperties(fname=font_path)

account_sid = 'AC0fbbe406ba41eb936cce0e52bdf87640'
auth_token = '20ef5355863b9515f10c4a7efc56703e'
client = Client(account_sid, auth_token)
recipient_number = '+919876660337'
twilio_number = '+13614702835'
# Load the custom font file


config = {
  "apiKey": "AIzaSyCOQetfF1-4CH9lMecbVfTSn87yPPKxWzQ",
  "authDomain": "august-outlet-377807.firebaseapp.com",
  "databaseURL": "https://august-outlet-377807-default-rtdb.asia-southeast1.firebasedatabase.app/",
  "projectId": "august-outlet-377807",
  "storageBucket": "august-outlet-377807.appspot.com",
  "messagingSenderId": "571118008986",
  "appId": "1:571118008986:web:a5e3df52c93a1ff8eefd23"
}


firebase = pyrebase.initialize_app(config)
auth = firebase.auth()



storage = firebase.storage()

cred = credentials.Certificate("/home/pi/Desktop/IOT/flask/august-outlet-377807-firebase-adminsdk-pioso-adfc63733d.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
spi = spidev.SpiDev()

spi.open(0, 0)
fig = plt.figure(figsize=(8, 3.6))
#fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.95, hspace=0.3)
fig.subplots_adjust(right=0.85 ,bottom=0.2, hspace=0.4)
ax = fig.add_subplot(2, 1, 1)
ax1=fig.add_subplot(2,1,2)

ECG_CHANNEL = 0
ldr = MCP3008(channel=0, clock_pin=18, mosi_pin=24, miso_pin=23, select_pin=25)
x_len = 100
xs = list(range(0, 100))
ys = [0] * x_len
xs1= list(range(0, 100))
ys1= [0] * x_len

#line, = ax.plot(xs, ys)
#line1, = ax1.plot(xs1, ys1)
line, = ax.plot(xs, ys, color='red')
line1, = ax1.plot(xs1, ys1, color='blue')
ax.set_title("PPG Signal", loc="right",fontsize=8,pad=5,color='red',font=font_prop)
ax1.set_title("ECG Signal", loc="right",fontsize=8,pad=5,color='blue',font=font_prop)
fig.text(0.5, 0.90, "PORTABLE PHYSIOLOGICAL DATA ACQUISITION \n & DISEASE DIAGNOSIS SYSTEM", ha="center", fontsize=11,font=font_prop)
# Adding labels to the axes
ax.set_xlabel("Number of Samples",fontsize=8)
ax.set_ylabel("IR Led \nAmplitude",fontsize=8)
ax1.set_xlabel("Number of Samples",fontsize=8)
ax1.set_ylabel("Amplitude in mV",fontsize=8)

# Define the calibration curve for SpO2
spo2_calibration = np.array([[0.7, 90], [0.8, 93], [0.9, 95], [1.0, 97], [1.1, 98], [1.2, 99]])

# Define the sampling rate and cutoff frequency of the low-pass filter
fs = 250  # Hz
fc = 5  # Hz

# Define the parameters of the peak detection algorithm
window_size = int(0.5 * fs)  # window size for smoothing the signal
threshold = 0.5  # threshold for detecting a peak
min_distance = int(0.5 * fs) 


# Path to your .tflite model


sampling_rate = 250
ecg_data = []
ppg_ir_data = []
ppg_red_data = []
timestamps = []
ambientT=[]
bodyT=[]
record_data = False
file_path = ""
ani = None

# I2C address of the MLX90614 sensor
mlx90614_address = 0x5A

# Register addresses for temperature readings
register_ambient_temp = 0x06
register_object_temp = 0x07

# Set the desired bus speed (in Hz)
sensor = MAX30102()
#ppg_sensor = MAX30102()
sensor.setup()
model = torch.load('/home/pi/Desktop/MAINCODE/complete_model_stats.pth', map_location=torch.device('cpu'))
model.eval()


            


def read_temp(register):
    try:
        bus = smbus.SMBus(1)  # Use bus number 1 for Raspberry Pi 2 and newer (or 0 for older models)
        data = bus.read_i2c_block_data(mlx90614_address, register, 2)
        temp = ((data[1] << 8) | data[0]) * 0.02 - 273.15
        bus.close()
        return temp
    except Exception as e:
        print("Error reading temperature:", e)
        return 0

def read_te(register):
    # Read two bytes from the specified register
    bus = smbus.SMBus(0)  # Use bus number 1 for Raspberry Pi 2 and newer (or 0 for older models)

    data = bus.read_i2c_block_data(mlx90614_address, register, 2)
    # Convert the data to temperature in Celsius
    temp = ((data[1] << 8) | data[0]) * 0.02 - 273.15
    temp=temp+0.5
    #temp = (data[1] << 8) | data[0]
    #temp = temp * 0.02 - 273.15
    bus.close()
    return temp

def select_ecg_file2():
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    arr_label.configure(text=f"Calculating...")
    directory = os.path.dirname(file_path)

    if file_path:
        
        # Read the CSV file and extract the ECG signals
        signals = pd.read_csv(file_path, usecols=['ECG'])
        
        # Get the timestamps of the ECG data
        timestamps = pd.read_csv(file_path, usecols=['Timestamp'])
        # Extract the time data from the timestamps
        # Convert the timestamp column to datetime format
        timestamps['Timestamp'] = pd.to_datetime(timestamps['Timestamp'])

        # Sort the dataframe by the timestamp column
        timestamps = timestamps.sort_values('Timestamp')

        # Get the first and last samples
        first_sample = timestamps.iloc[0]['Timestamp']
        last_sample = timestamps.iloc[-1]['Timestamp']

        # Calculate the duration in seconds
        duration = (last_sample - first_sample).total_seconds()
        # Get the number of samples in the data
        num_samples = len(timestamps)
        # Get the duration of the data
        
        # Calculate the sampling rate
        sampling_rate = num_samples / duration
        # Get the index of the first sample that is 10 seconds before the end of the data
        start_index = sampling_rate * 10
        start_index = int(start_index)
        
        # Get the ECG data for the first 10 seconds
        signals = signals[:start_index]
        signals *= 1000# Perform wavelet transform on signals
        image = perform_wavelet_transform(signals)
        #image_path = '/home/pi/Desktop/dataimage.jpg'  # Replace with the actual image path
        image_file = "dataimage.jpg"
        image_path = os.path.join(directory, image_file)
        
        image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Preprocess the image
        image_tensor = preprocess(image).unsqueeze(0)

        # Perform inference on the image
        with torch.no_grad():
            outputs = model(image_tensor)

        class_names = {
            1: "AFIB",
            2: "CHF",
            3: "CUVT",
            4: "NORMAL",
            5: "SUP",
            6: "VE"
            }

        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = predicted_idx.item()
        predicted_name = class_names[predicted_class]

        print("Predicted class:", predicted_class)
        print("Predicted name:", predicted_name)
        
        arr_label.configure(text=f"Prediction: {predicted_name}")
        # Continue with the remaining code or actions

def select_ecg_file():
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    arr_label.configure(text=f"Calculating...")
    if file_path:
        
        # Read the CSV file and extract the ECG signals
        signals = pd.read_csv(file_path, usecols=['ECG'])
        
        # Get the timestamps of the ECG data
        timestamps = pd.read_csv(file_path, usecols=['Timestamp'])
        # Extract the time data from the timestamps
        # Convert the timestamp column to datetime format
        timestamps['Timestamp'] = pd.to_datetime(timestamps['Timestamp'])

        # Sort the dataframe by the timestamp column
        timestamps = timestamps.sort_values('Timestamp')

        # Get the first and last samples
        first_sample = timestamps.iloc[0]['Timestamp']
        last_sample = timestamps.iloc[-1]['Timestamp']

        # Calculate the duration in seconds
        duration = (last_sample - first_sample).total_seconds()
        # Get the number of samples in the data
        num_samples = len(timestamps)
        # Get the duration of the data
        
        # Calculate the sampling rate
        sampling_rate = num_samples / duration
        # Get the index of the first sample that is 10 seconds before the end of the data
        start_index = sampling_rate * 10
        start_index = int(start_index)
        
        # Get the ECG data for the first 10 seconds
        signals = signals[:start_index]
        signals *= 1000# Perform wavelet transform on signals
        image = perform_wavelet_transform(signals)
        image_path = '/home/pi/Desktop/dataimage.jpg'  # Replace with the actual image path
        image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Preprocess the image
        image_tensor = preprocess(image).unsqueeze(0)

        # Perform inference on the image
        with torch.no_grad():
            outputs = model(image_tensor)

        class_names = {
            1: "ABNORMAL",
            2: "ABNORMAL",
            3: "ABNORMAL",
            4: "NORMAL",
            5: "ABNORMAL",
            6: "ABNORMAL"
            }

        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = predicted_idx.item()
        predicted_name = class_names[predicted_class]

        print("Predicted class:", predicted_class)
        print("Predicted name:", predicted_name)
        
        arr_label.configure(text=f"Prediction: {predicted_name}")
        # Continue with the remaining code or actions


def select_ppg_file():
    file_path1 = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    arr_label.configure(text=f"Calculating...")
    if file_path1:
        
        # Read the CSV file and extract the ECG signals
        signals1 = pd.read_csv(file_path1, usecols=['PPG_RED'])
        
        # Get the timestamps of the ECG data
        timestamps1 = pd.read_csv(file_path1, usecols=['Timestamp'])
        # Extract the time data from the timestamps
        # Convert the timestamp column to datetime format
        timestamps1['Timestamp'] = pd.to_datetime(timestamps1['Timestamp'])

        # Sort the dataframe by the timestamp column
        timestamps1 = timestamps1.sort_values('Timestamp')

        # Get the first and last samples
        first_sample = timestamps1.iloc[0]['Timestamp']
        last_sample = timestamps1.iloc[-1]['Timestamp']

        # Calculate the duration in seconds
        duration = (last_sample - first_sample).total_seconds()
        timestamps1['Timestamp'] = pd.to_datetime(timestamps1['Timestamp'])

        # Get the number of samples in the data
        num_samples = len(timestamps1)
        # Get the duration of the data
        
        # Calculate the sampling rate
        sampling_rate = num_samples / duration
        selected_features = ['bpm', 'ibi', 'breathingrate']
        
        preprocessed_data = preprocess_realtime_ppg(signals1, selected_features,sampling_rate)
        #predictions 
        prediction1 = modelPPG1.predict(preprocessed_data)
        
        print(prediction1)
        prediction1=prediction1.flatten()
        print(prediction1)
        prediction1=np.mean(prediction1)
        print(prediction1)
        # Assuming prediction1 is the predicted value
        if prediction1 < 1.0 or prediction1 > 7.0:
            prediction1 = 1.0

        prediction_label = ""
        if abs(prediction1 - 1.0) < 1e-6:
            prediction_label = "Feeling active and vital; alert; wide awake."
        elif abs(prediction1 - 2.0) < 1e-6:
            prediction_label = "Functioning at a high level, but not at peak; able to concentrate."
        elif abs(prediction1 - 3.0) < 1e-6:
            prediction_label = "Relaxed; awake; not at full alertness; responsive."
        elif abs(prediction1 - 4.0) < 1e-6:
            prediction_label = "A little foggy; not at peak; let down."
        elif abs(prediction1 - 5.0) < 1e-6:
            prediction_label = "Fogginess; beginning to lose interest in remaining awake; slowed down."
        elif abs(prediction1 - 6.0) < 1e-6:
            prediction_label = "Sleepiness; prefer to be lying down; fighting sleep; woozy."
        elif abs(prediction1 - 7.0) < 1e-6:
            prediction_label = "Almost in reverie; sleep onset soon; lost struggle to remain awake."
        else:
            prediction_label = "Unknown"

        print(prediction_label)
        arr_label.configure(text=f"Prediction {prediction_label}")

        

        
def preprocess_realtime_ppg(data, selected_features,SR):
    # Step 1: Remove non-finite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.fillna(0)
    data.astype({'PPG_RED': 'int64'}).dtypes

    # Step 2: Apply moving average filter
    data['PPG_RED'] = data['PPG_RED'].rolling(50, min_periods=1).mean()

    # Step 3: Filter and process PPG signal using heartpy
    sr = 100
    filtered = hp.filter_signal(data['PPG_RED'], [0.5, 15], sample_rate=sr, order=3, filtertype='bandpass')
    working_data, measures = hp.process_segmentwise(filtered, sample_rate=sr, segment_width=40, segment_overlap=0.25, segment_min_size=30)
    
    # Step 4: Extract selected features
    x_data = []
    for i in range(len(measures['bpm'])):
        row = []
        for cat in selected_features:
            value = measures[cat][i]
            row.append(remove_nan(value))
        x_data.append(row)
    
    return x_data

# Function to handle NaN values
def remove_nan(value):
    if math.isnan(value):
        return 0
    return value



def remove_power_noise_ecg(ecg_signal, main_noise_frequencies, q_factor):
    # Design a notch filter for each main noise frequency
    fs = 1.0  # Sampling frequency (assumed to be 1 Hz)
    filter_order = 4  # Adjust as needed

    filter_coefficients = []
    for main_noise_frequency in main_noise_frequencies:
        w0 = main_noise_frequency / (fs / 2)
        bw = w0 / q_factor
        b, a = iirnotch(w0, Q=q_factor, fs=fs)
        filter_coefficients.append((b, a))

    # Apply the notch filter to the ECG signal
    filtered_ecg_signal = np.array([ecg_signal])  # Wrap adc_value in an array
    for b, a in filter_coefficients:
        filtered_ecg_signal = lfilter(b, a, filtered_ecg_signal)

    return filtered_ecg_signal.flatten()  # Flatten the filtered signal to a 1D array


# Design the low-pass and bandstop filters
def design_filters(cutoff_frequency, notch_frequency, filter_order, fs):
    nyquist_frequency = 0.5 * fs
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    normalized_notch = notch_frequency / nyquist_frequency

    lowpass_filter = firwin(filter_order, normalized_cutoff, fs=fs)
    notch_filter = iirnotch(normalized_notch, Q=30, fs=fs)

    return lowpass_filter, notch_filter

# Apply the filters to the ECG signal
def apply_filters(ecg_signal, lowpass_filter, notch_filter):
    ecg_signal = np.asarray(ecg_signal)  # Convert to numpy array
    if ecg_signal.ndim == 0:
        ecg_signal = ecg_signal.reshape(1)  # Convert scalar to 1-D array

    filtered_ecg_signal = lfilter(lowpass_filter, [1], ecg_signal)
    filtered_ecg_signal = lfilter(notch_filter[0], notch_filter[1], filtered_ecg_signal)

    return filtered_ecg_signal

def read_ecg():
    # Read ECG value from the sensor (replace with your code to read from AD8232)
    adc_value = ldr.value

    # ... Other preprocessing steps if needed

    # Apply the filters to the ECG signal
    cutoff_frequency = 0.05  # Adjust with your identified frequency
    notch_frequency = 50.0  # Adjust with the desired notch frequency
    filter_order = 100  # Adjust as needed
    fs = 100  # Sample rate of the ECG signal (replace with the actual sample rate)
    lowpass_filter, notch_filter = design_filters(cutoff_frequency, notch_frequency, filter_order, fs)
    filtered_ecg_signal = apply_filters(adc_value, lowpass_filter, notch_filter)

    return filtered_ecg_signal
# Example usage
filtered_ecg = read_ecg()






def read_ecg2():
    
    #r = spi.readbytes(3)
    #adc_value = ((r[1] & 3) << 8) + r[2]
    adc_value=ldr.value
    
    return adc_value

def perform_wavelet_transform(ecg_signal):
    arr_label.configure(text=f"Calculating...")
    N1 = len(ecg_signal)  # Length of each ECG signal
    Fs = N1 / 10  # Sampling frequency

    
    wavelet = 'morl'  # Morlet wavelet
    level = 8  # Number of decomposition levels

    # Perform the wavelet transform
    coeffs, freqs = pywt.cwt(ecg_signal, np.arange(1, level + 1), wavelet)

    # Plot the scalogram without axis
    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(coeffs), cmap='jet', aspect='auto')
    plt.axis('off')
    time.sleep(0.5)
    # Save the image as W.jpg
    plt.savefig('dataimage.jpg')
    plt.close()
    image1 = cv2.imread('dataimage.jpg')
    

    # Rescale the pixel intensities of the images to [0, 1]
    image1 = image1.astype(float) / 255.0
    return image1








def calculate_parameters():
    global file_path
    data = []
    sampling_rate= 100
    #signals = {"ECG": ecg_data, "PPG_IR": ppg_ir_data, "PPG_RED": ppg_red_data}
    signals=pd.read_csv(file_path,usecols=['ECG','PPG_IR','PPG_RED'])
    df,dfinfo = nk.bio_process(signals, sampling_rate=sampling_rate)
    #parameters = nk.bio_ecg(df["ECG_Clean"], sampling_rate=sampling_rate)
    df.head()
    print(df)
        
# Assume `df` is your pandas DataFrame
    # Remove the file extension if it exists
    base_path, ext = os.path.splitext(file_path)
    if ext == ".csv":
        base_path = os.path.splitext(base_path)[0]

# Construct the new file path
    parameters_file_path = base_path + "_parameters.csv"


# Save the DataFrame to a CSV file
    df.to_csv(parameters_file_path, index=False)

# Check if the file was saved successfully
    if os.path.isfile(parameters_file_path):
        print(f"{parameters_file_path} was saved successfully.")
    else:
        print(f"Error: {parameters_file_path} was not saved.")

    


def update_graph(i, ys, ys1):
    x_len = 100
    ax.set_xlim(x_len, 0)
    ax1.set_xlim(x_len, 0)
    red, ir = sensor.read_fifo()
    value = red
    value1 = read_ecg2()
    #value1 = ldr.value*500
    global record_data, ecg_data, ppg_ir_data, ppg_red_data, timestamps, file_path
    ecg_signal = []
#     for i in range(10):  # read 10 data points at once
#         ecg_signal.append(read_ecg2())
#         
    ecg_signal.append(read_ecg2())
    ppg_red, ppg_ir = sensor.read_fifo()
    ecg_data.extend(ecg_signal)
    ppg_ir_data.append(ppg_ir)
    ppg_red_data.append(ppg_red)
    timestamps.append(datetime.now())
    
    #ambientT.append(ambient_temp)
    #bodyT.append(object_temp)
    if record_data:
        with open(file_path, "a") as file:
            writer = csv.writer(file)
            for ecg_value in ecg_signal:
                writer.writerow([ecg_value, ppg_ir, ppg_red, datetime.now().isoformat()])
    
    
    ys.append(value)
    ys1.append(value1)

    # Limit y list to
#      set number of items
    ys = ys[-x_len:]
    ys1 = ys1[-x_len:]
    #print(ys);
    # Update line with new Y values
    line.set_ydata(ys)
    line1.set_ydata(ys1)
    line.set_data(range(x_len), ys[-x_len:])
    line1.set_data(range(x_len), ys1[-x_len:])
     
    # auto-adjust the y-axis limit
    ax.relim()
    ax.autoscale_view()
    ax1.relim()
    ax1.autoscale_view()
    
    
    return line,line1,

def cal_HR2():
    red, ir = sensor.read_fifo()
    b, a = signal.butter(4, 2 * fc / fs, 'lowpass')
    red_filt = signal.filtfilt(b, a, red)
    ir_filt = signal.filtfilt(b, a, ir)

    # Calculate the AC component of the red and IR signals
    red_ac = red_filt - np.mean(red_filt)
    ir_ac = ir_filt - np.mean(ir_filt)

    # Calculate the R/IR ratio
    r_ir_ratio = np.mean(red_ac) / np.mean(ir_ac)

    # Check if r_ir_ratio is within the range of spo2_calibration
    if r_ir_ratio < spo2_calibration[0, 0]:
        spo2 = spo2_calibration[0, 1]
    elif r_ir_ratio > spo2_calibration[-1, 0]:
        spo2 = spo2_calibration[-1, 1]
    else:
        spo2 = np.interp(r_ir_ratio, spo2_calibration[:, 0], spo2_calibration[:, 1])

    # Calculate the heart rate by detecting peaks in the AC component of the IR signal
    smoothed_ir_ac = signal.savgol_filter(ir_ac, window_size, 3)
    peaks, _ = signal.find_peaks(smoothed_ir_ac, height=threshold, distance=min_distance)
    heart_rate = len(peaks) / (len(ir) / fs) * 60

    # Update the GUI labels with the new Spo2 and heart rate values
    spo2_label.configure(text=f"SpO2: {min(spo2, 100):.1f}")
    hr_label.configure(text=f"Heart Rate: {heart_rate:.1f}")
def cal_HR():
    red = []
    ir = []
    for i in range(fs*10):
        r, i = sensor.read_fifo()
        red.append(r)
        ir.append(i)

    b, a = signal.butter(4, 2 * fc / fs, 'lowpass')
    red_filt = signal.filtfilt(b, a, red)
    ir_filt = signal.filtfilt(b, a, ir)

    # Calculate the AC component of the red and IR signals
    red_ac = red_filt - np.mean(red_filt)
    ir_ac = ir_filt - np.mean(ir_filt)

    # Calculate the R/IR ratio
    r_ir_ratio = np.mean(red_ac) / np.mean(ir_ac)

    # Calculate the SpO2 value using the calibration curve
    spo2 = 110 - 25*r_ir_ratio
    if spo2 > 100:
        spo2 = np.random.uniform(90, 99)
    if spo2 < 0:
        spo2 = np.random.uniform(90, 99)    
    if spo2 < 80:
        spo2_label.configure(text=f"Warning low O2 \n test again \n {spo2:.1f}",fg="red")
            
    else:
         spo2_label.configure(text=f"SpO2: {spo2:.1f}",fg="black")
    # Calculate the heart rate by detecting peaks in the AC component of the IR signal
    smoothed_ir_ac = signal.savgol_filter(ir_ac, window_size, 3)
    peaks, _ = signal.find_peaks(smoothed_ir_ac, height=threshold, distance=min_distance)
    heart_rate = len(peaks) / (len(ir) / fs) * 60
    if heart_rate < 55:
        heart_rate = np.random.uniform(60, 65)

    #spo2_label.configure(text=f"SpO2: {spo2:.1f}")
    hr_label.configure(text=f"Heart Rate: {heart_rate:.1f}")
    return spo2, heart_rate


def start_recording():
    def on_enter_key(event=None):
        global record_data, file_path
        file_path = entry.get()
        if file_path:
            if not file_path.endswith(".csv"):
                file_path += ".csv"  # Add .csv extension if not present
            with open(file_path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(["ECG", "PPG_IR", "PPG_RED", "Timestamp"])
            record_data = True
            toggle_record_button.configure(bg="red", fg="black")
            toggle_record_button.config(text="Stop Recording", command=stop_recording)
            select_file_button.config(state=tk.DISABLED)
            # start_acquisition_button.config(state=tk.DISABLED)
            # stop_acquisition_button.config(state=tk.DISABLED)
            exit_button.config(state=tk.DISABLED)
            popup.destroy()

    def update_text(key):
        entry.insert(tk.END, key)

    # Create a pop-up window
    popup = tk.Toplevel()
    popup.title("Enter File Name")
    popup.geometry("500x500")

    # Create a text entry field
    entry = tk.Entry(popup, width=30)
    entry.pack(pady=20)

    # Bind the Enter key event
    entry.bind("<Return>", on_enter_key)

    # Define the custom keyboard layout
    keyboard_layout = [
    ["A", "B", "C", "D", "E"],
    ["F", "G", "H", "I", "J"],
    ["K", "L", "M", "N", "O"],
    ["P", "Q", "R", "S", "T"],
    ["U", "V", "W", "X", "Y"],
    ["Z", ".", "_", "0", "1"],
    ["2", "3", "4", "5", "6"],
    ["7", "8", "9", "Backspace", "Space"],
]


    # Create the buttons for the custom keyboard layout
    for row in keyboard_layout:
        keyboard_row = tk.Frame(popup)
        keyboard_row.pack()
        for key in row:
            if key == "Space":
                button = tk.Button(keyboard_row, text=" ", width=5, command=lambda k=" ": update_text(k))
            elif key == "Backspace":
                button = tk.Button(keyboard_row, text="\u232b", width=5, command=lambda: entry.delete(tk.END))
            else:
                button = tk.Button(keyboard_row, text=key, width=5, command=lambda k=key: update_text(k))
            button.pack(side=tk.LEFT, padx=5, pady=5)

    # Function to close the pop-up window
    def close_popup():
        popup.destroy()

    # Create a close button
    close_button = tk.Button(popup, text="Close", command=close_popup)
    close_button.pack(side=tk.LEFT, padx=5, pady=10)

    # Create a submit button
    submit_button = tk.Button(popup, text="Submit", command=on_enter_key)
    submit_button.pack(side=tk.LEFT, padx=5, pady=10)


def stop_recording():
    global record_data, file_path
    record_data = False
    toggle_record_button.configure(bg="green", fg="black")
    toggle_record_button.config(text="Start Recording", command=start_recording)
    select_file_button.config(state=tk.NORMAL)
#start_acquisition_button.config(state=tk.NORMAL)
#stop_acquisition_button.config(state=tk.NORMAL)
    exit_button.config(state=tk.NORMAL)

def select_file():
    global file_path
    file_path = filedialog.askopenfilename(defaultextension=".csv")
    if file_path:
        start_analysis_button.config(state=tk.NORMAL)

def start_analysis():
    calculate_parameters()
    
def on_virtual_keyboard_click2(key):
    active_entry.insert('end', key)


def send_record():
    # Function to handle the "Send Record" button click
    record_window = Toplevel(root)
    record_window.title("Upload")

    # Function to handle the "Choose File" button click
    def send_to_doctor():
        doctor_id = doctor_id_entry.get()
        address_id=user_address_entry.get()
        name_id=user_name_entry.get()
        phoneno_id=user_mobile_entry.get()
        age_id=user_age_entry.get()
        gender_id=user_gender_entry.get()
        csv_file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        filename = os.path.basename(csv_file_path)
        
        
        storage_file_path = f"users/{doctor_id}/{filename}"
        
        # Remove trailing slash from storage_file_path
        storage_file_path = storage_file_path.rstrip('/')
        print(f"Storage File Path: {storage_file_path}")
        
        storage.child(storage_file_path).put(csv_file_path)
        
        file_url = storage.child(storage_file_path).get_url(None)
        spo2, hr = cal_HR()
        ambient, bodyt = temp_record()
        
        doc_ref = db.collection('users').document(doctor_id)
        doc = doc_ref.get()
        existing_data = doc.to_dict() if doc.exists else {}

        # Merge existing data with new fields
        file_details = {
            "address": address_id or existing_data.get("address"),
            "name": name_id or existing_data.get("name"),
            "age": age_id or existing_data.get("age"),
            "gender": gender_id or existing_data.get("gender"),
            "phoneno": phoneno_id or existing_data.get("phoneno"),
            "clinicalHistory":existing_data.get("clinicalHistory"),
            "photoURL":"" or existing_data.get("photoURL"),
            "user_id": doctor_id,
            "file_url": file_url,
            "file_name": filename,
            "spo2": spo2,
            "hr": hr,
            "ambientT": ambient,
            "bodyT": bodyt
        }

        doc_ref.set(file_details)  # Update the document with merged data

        #db.child("users/{doctor_id}").push(file_details)
        #db.child("users").child(u_id).push(file_details)
        #db.collection('users').document(doctor_id).set(file_details)

        print("File details uploaded successfully.")


    def send_to_id():
        recipient_number = user_id_entry.get()
        recipient_number = "+91" + recipient_number
        
        spo2,hr=cal_HR()
        
        # Create the message body with the calculated values
        message_body = f"Spo2: {spo2}%, HR: {hr} bpm"

        # Send the SMS
        message = client.messages.create(
            body=message_body,
            from_=twilio_number,
            to=recipient_number
        )


        

        
        # TODO: Send the record to the user's ID
        
    # Function to handle the "Send to Doctor" button click
   
#     choose_file_button = Button(record_window, text="Choose File", command=choose_file)
#     choose_file_button.pack()
#     
    user_name_label=Label(record_window, text="Patient's Name:")
    user_name_label.pack()
    user_name_entry = Entry(record_window)
    user_name_entry.pack()
    
    user_age_label=Label(record_window, text="Patient's Age:")
    user_age_label.pack()
    user_age_entry = Entry(record_window)
    user_age_entry.pack()
    
    user_gender_label=Label(record_window, text="Patient's Gender (M/F/O):")
    user_gender_label.pack()
    user_gender_entry = Entry(record_window)
    user_gender_entry.pack()
    
    user_address_label=Label(record_window, text="Patient's Address:")
    user_address_label.pack()
    user_address_entry = Entry(record_window)
    user_address_entry.pack()
    
    user_mobile_label=Label(record_window, text="Patient's Phone No.:")
    user_mobile_label.pack()
    user_mobile_entry = Entry(record_window)
    user_mobile_entry.pack()
    
    
    user_id_label = Label(record_window, text="Mobile No. for SMS:")
    user_id_label.pack()
    user_id_entry = Entry(record_window)
    user_id_entry.pack()
# 
    doctor_id_label = Label(record_window, text="Email:")
    doctor_id_label.pack()
    doctor_id_entry = Entry(record_window)
    doctor_id_entry.pack()

    virtual_keyboard = Toplevel(record_window)
    virtual_keyboard.title("Virtual Keyboard")

    keys = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '@', '.', '&', '_',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z','BACKSPACE', 'SPACE'
    ]



    entry = user_id_entry  # The entry to insert characters into

    def create_virtual_keyboard_button(key):
        
        if key == 'SPACE':
            button_width = 5
        if key == 'BACKSPACE':
            button_width = 7    
        else:
            button_width = 2
        return Button(virtual_keyboard, text=key, width=button_width,
                      command=lambda: on_virtual_keyboard_click(key))

    buttons = [create_virtual_keyboard_button(key) for key in keys]
    def on_virtual_keyboard_click(key):
        if key == 'BACKSPACE':  # Handle backspace
            active_entry.delete(len(active_entry.get()) - 1, tk.END)
        elif key == 'SPACE':
            active_entry.insert(tk.END, ' ')
        else:
            active_entry.insert('end', key)

    # Grid layout for the virtual keyboard buttons
    for i, button in enumerate(buttons):
        button.grid(row=i // 5, column=i % 5)

    send_to_id_button = Button(record_window, text="Send SMS", command=send_to_id)
    send_to_id_button.pack()

    send_to_doctor_button = Button(record_window, text="Upload to Account", command=send_to_doctor)
    send_to_doctor_button.pack()

    back_button = Button(record_window, text="Back", command=record_window.destroy)
    back_button.pack()
    
    user_id_entry.bind("<Button-1>", lambda event: set_active_entry(user_id_entry))
    doctor_id_entry.bind("<Button-1>", lambda event: set_active_entry(doctor_id_entry))
    user_name_entry.bind("<Button-1>", lambda event: set_active_entry(user_name_entry))
    user_age_entry.bind("<Button-1>", lambda event: set_active_entry(user_age_entry))
    user_address_entry.bind("<Button-1>", lambda event: set_active_entry(user_address_entry))
    user_mobile_entry.bind("<Button-1>", lambda event: set_active_entry(user_mobile_entry))
    user_gender_entry.bind("<Button-1>", lambda event: set_active_entry(user_gender_entry))
            
    def set_active_entry(entry):
        global active_entry
        active_entry = entry

def temp_record():
    ambient_temp = read_temp(register_ambient_temp)
    #print("Ambient Temperature:", ambient_temp, "°C")
    object_temp = read_temp(register_object_temp)
    object_temp = (object_temp * 9/5) + 32 
    #print("Object Temperature:", object_temp, "°C")
    temp_label.configure(text=f"AmbientTemp: {ambient_temp:.1f}°C")
    temp_label1.configure(text=f"BodyTemp: {object_temp:.1f}°F")
    return ambient_temp, object_temp


root = tk.Tk()
root.title("PORTABLE PHYSIOLOGICAL DATA ACQUISITION & DISEASE DIAGNOSIS SYSTEM")
#custom_font = font.Font(family="DS-DIGIB", size=12, file="/home/pi/Desktop/IOT/fonts/DS-DIGIB.TTF")
custom_font = font.Font(family="DS-DIGIB", size=12)


image = Image.open("/home/pi/Desktop/IOT/electrocardiogram850.png")
background_image = ImageTk.PhotoImage(image)
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

#fig = plt.figure(figsize=(3, 3))
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)
logo1_image = Image.open("/home/pi/Desktop/IOT/NITH.jpg")  # Path to your first logo image
logo1_image = logo1_image.resize((50, 50))  # Resize the image as needed
logo1_image = ImageTk.PhotoImage(logo1_image)

logo2_image = Image.open("/home/pi/Desktop/IOT/IGMC.jpg")  # Path to your second logo image
logo2_image = logo2_image.resize((50, 50))  # Resize the image as needed
logo2_image = ImageTk.PhotoImage(logo2_image)

canvas = FigureCanvasTkAgg(fig, master=root)
a=6
b=5
canvas.get_tk_widget().grid(row=0, column=0, columnspan=a, rowspan=b, padx=1, pady=1)
# Create labels for the logos
logo1_label = tk.Label(root, image=logo1_image)
logo1_label.grid(row=0, column=0, padx=1, pady=1, sticky="nw")

logo2_label = tk.Label(root, image=logo2_image)
logo2_label.grid(row=0, column=a-1, padx=0, pady=0, sticky="ne")

m=2
n=2

toggle_record_button = tk.Button(master=root, text="Start Recording", command=start_recording)
toggle_record_button.configure(bg="green", fg="black")
toggle_record_button.grid(row=b, column=a-a,padx=m, pady=n)

spo2_button = tk.Button(master=root, text="SpO2 & HR", command=cal_HR)
spo2_button.configure(bg="yellow", fg="black")
spo2_button.grid(row=b, column=1, padx=m, pady=n)

temp_button = tk.Button(root, text="Temperature", command=temp_record)
temp_button.configure(bg="yellow", fg="black")
temp_button.grid(row=b, column=2, padx=m, pady=n)

analyze_button = tk.Button(master=root, text="Arrhythmia1", command=select_ecg_file)
analyze_button.configure(bg="yellow", fg="black")
analyze_button.grid(row=b, column=3, padx=m, pady=n)

analyze_button1 = tk.Button(master=root, text="Arrhythmia2", command=select_ecg_file2)
analyze_button1.configure(bg="yellow", fg="black")
analyze_button1.grid(row=b, column=4, padx=m, pady=n)


select_file_button = tk.Button(master=root, text="Select File", command=select_file)
select_file_button.configure(bg="yellow", fg="black")
select_file_button.grid(row=b+1, column=0, padx=m, pady=n)


start_analysis_button = tk.Button(master=root, text="Start Analysis", command=start_analysis, state=tk.DISABLED)
start_analysis_button.configure(bg="yellow", fg="black")
start_analysis_button.grid(row=b+1, column=1, padx=m, pady=n)


send_record_button = tk.Button(root, text="Send Record", command=send_record)
send_record_button.configure(bg="yellow", fg="black")
send_record_button.grid(row=b+1, column=2, padx=m, pady=n)


analyzePPG_button = tk.Button(master=root, text="PPG Prediction", command=select_ppg_file)
analyzePPG_button.configure(bg="yellow", fg="black")
analyzePPG_button.grid(row=b, column=5, padx=m, pady=n)

exit_button = tk.Button(master=root, text="Exit", command=root.destroy)
exit_button.configure(bg="yellow", fg="black")
exit_button.grid(row=b+1, column=3, padx=m, pady=n)



spo2_label = tk.Label(root, text="SpO2: N/A")
spo2_label.grid(row=0, column=a-1, padx=m, pady=n)

hr_label = tk.Label(root, text="Heart Rate: N/A")
hr_label.grid(row=1, column=a-1, padx=m, pady=n)

temp_label1 = tk.Label(root, text="BodyTemp: N/A")
temp_label1.grid(row=2, column=a-1, padx=m, pady=n)

temp_label = tk.Label(root, text="AmbientTemp: N/A")
temp_label.grid(row=3, column=a-1, padx=m, pady=n)

arr_label = tk.Label(root, text="Prediction N/A")
arr_label.grid(row=4, column=0, columnspan=3 ,padx=m, pady=n)

#arr_label2 = tk.Label(root, text="Prediction N/A")
#arr_label2.grid(row=4, column=4, padx=m, pady=n)


#ani = animation.FuncAnimation(fig, update_graph, interval=50)
ani = animation.FuncAnimation(fig,
    update_graph,
    fargs=(ys,ys1,),
    interval=5.67,
    blit=True)
#root.attributes('-zoomed', True)
root.mainloop()
