import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to ignore tensorflow warnings
import warnings
import numpy as np
import librosa as lb
import librosa.display
import tensorflow_hub as hub
# import tensorflow as tf
import noisereduce as nr
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from tensorflow import keras
from keras.utils import load_img, img_to_array
# from keras.preprocessing import image     # THIS IS FOR LOWER VERSIONS OF TENSORFLOW 2.0
from scipy.signal import argrelextrema
from natsort import os_sorted
import pandas as pd


# GLOBAL PARAMETERS AND FILE PATHS #
# MAIN_FOLDER = os.getcwd()
MAIN_FOLDER = "D:\\Aj\\BirdClassification\\RPi"
IMG_SIZE = 224
FRAME_SIZE = 2048
HOP_SIZE = 512
SR = 22050
N_MELS = 128
DURATION = 5.0
LOWCUT = 1000
HIGHCUT = 11000
EVENT_THRESHOLD = 0.03 # Threshold for event detection; original value is 0.25.
seconds_per_buffer = 10
buffer = 22050 * seconds_per_buffer # buffer window of 5 seconds

# sample = "/home/ajdevera/Desktop/Bird Classifier/2022-10-03_Lunchtime_Recording1.wav"
sample = "D:\\Aj\\BirdClassification\\FROM RPI RECORDINGS\\Recordings\\Oct 27\\2022-10-27_EarlyMorning_Recording1.wav"
# sample = "D:\\Aj\\BirdClassification\\FROM RPI RECORDINGS\\Recordings\\Oct 21\\test\\filtered3.wav"

warnings.filterwarnings('ignore')

# species = ['AsianGlossyStarling', 'Black-crownedNightHeron', 'Black-napedOriole', 'Blue-headedFantail', 'Blue-tailedBee-eater', 'BrownShrike', 'ChestnutMunia', 'CollaredKingfisher', 'EurasianTreeSparrow', 'Grey-backedTailorbird', 'GreyWagtail', 'MangroveBlueFlycatcher', 'Olive-backedSunbird', 'PhilippineMagpie-Robin', 'PhilippinePiedFantail', 'PiedBushChat', 'Red-keeledFlowerpecker', 'Rufous-crownedBee-eater', 'White-breastedWaterhen', 'Yellow-ventedBulbul']
species = ['AsianGlossyStarling', 'Black-crownedNightHeron', 'Black-napedOriole', 'Blue-tailedBee-eater', 'BrownShrike', 'ChestnutMunia', 'CollaredKingfisher', 'EurasianTreeSparrow', 'Grey-backedTailorbird', 'GreyWagtail', 'MangroveBlueFlycatcher', 'Olive-backedSunbird', 'PhilippineMagpie-Robin', 'PhilippinePiedFantail', 'PiedBushChat', 'Red-keeledFlowerpecker', 'White-breastedWaterhen', 'Yellow-ventedBulbul', 'ZebraDove']
print(species)

# CHOOSE CNN MODEL
# 1 = CUSTOM CNN; 2 = EFFICIENTNETV2;
model_select = 1
if model_select == 1:
    MODEL_PATH = "D:\\Aj\\BirdClassification\\FROM RPI RECORDINGS\\custom_model_CNN new dataset no noisereduced2.h5"
    IMG_SIZE = 224
elif model_select == 2:
    MODEL_PATH = "D:\\Aj\\BirdClassification\\FROM RPI RECORDINGS\\efficientnetv2_b2_imagenet1kefficientnetv2_CNN new dataset no noisereduced.h5"
    IMG_SIZE = 260
elif model_select == 3:
    MODEL_PATH = "D:\\Aj\\BirdClassification\\FROM RPI RECORDINGS\\MobileNetV3mobilenetv3_CNN new dataset no noisereduced.h5"
    IMG_SIZE = 224

SPECTROGRAM_PATH = "D:\\Aj\\BirdClassification\\FROM RPI RECORDINGS\\Spectrograms 0.03 threshold"
if not os.path.exists(SPECTROGRAM_PATH):
    os.mkdir(SPECTROGRAM_PATH)

filename, _ = os.path.splitext(sample)
filename = filename.split("\\")
filename = filename[-1]
WAVFILE_SPECTROGRAM = os.path.join(SPECTROGRAM_PATH, filename)
print(WAVFILE_SPECTROGRAM)
if not os.path.exists(WAVFILE_SPECTROGRAM):
    os.mkdir(WAVFILE_SPECTROGRAM)


def load_wavfile(wavfile_path, minute):
    signal, _ = lb.load(wavfile_path, offset=minute, duration=60)
    return signal


def butter_bandpass(lowcut, highcut, fs, order=7):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=7):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def load_cnn_model(modelpath):
    if model_select == 1:
        model = keras.models.load_model(modelpath)
    else:
        model = keras.models.load_model(modelpath, custom_objects={'KerasLayer':hub.KerasLayer})
    return model


def load_audio(filepath, offset):
    off_set = OFFSET + offset - seconds_per_buffer
    audio, _ = lb.load(filepath, offset=off_set, duration=DURATION)
    audio = butter_bandpass_filter(audio, LOWCUT, HIGHCUT, SR)
    # audio = nr.reduce_noise(y=audio, sr=SR)
    return audio


def extract_logmel_spectrogram(audio):
    mel = lb.feature.melspectrogram(audio, sr=SR, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=N_MELS)
    log_mel = lb.power_to_db(mel)
    return log_mel


def save_spectrogram(logmel_spec, time_of_event, minute):
    filename, _ = os.path.splitext(sample)
    # time_of_event variable is the last second of the buffer e.g. at 8:47, an event was detected. time_of_event is 8:50
    # such that the buffer was from 8:45 to 8:50

    # If-elif-else statement is for filename correction.
    if time_of_event <= 10:
        # If :00 to :10, filename is corrected from "10 0_10 5.png" to "10 00_10 05.png"
        if time_of_event < 10:
            temp = f"Time_{int(minute) - 1} 0{int(time_of_event) - int(DURATION)}_{int(minute) - 1} 0{time_of_event}.png"
        elif time_of_event == 10:
            temp = f"Time_{int(minute) - 1} 0{int(time_of_event) - int(DURATION)}_{int(minute) - 1} {time_of_event}.png"
    elif time_of_event == 60:
        # fiename is corrected from "10 55_10 60.png" to "10 55_11 00.png"
        if minute == 60:
            temp_minute = minute
        else:
            temp_minute = minute + 1
        time_of_event2 = "00"
        temp = f"Time_{int(temp_minute) - 1} {int(time_of_event) - int(DURATION)}_{int(temp_minute)} {time_of_event2}.png"
    else:
        temp = f"Time_{int(minute) - 1} {int(time_of_event) - int(DURATION)}_{int(minute) - 1} {time_of_event}.png"
    temporary_filename = os.path.join(WAVFILE_SPECTROGRAM, temp)
    fig = lb.display.specshow(logmel_spec, cmap='jet', sr=SR)
    fig.figure.savefig(temporary_filename, bbox_inches='tight', pad_inches=0.0)
    fig.remove()
    plt.close()


def preprocess_and_save_spectrogram(wavfile_path, time_of_event, minute):
    audio = load_audio(wavfile_path, time_of_event)
    audio = lb.effects.preemphasis(audio)
    # if np.max(audio) < 0.5 and np.max(audio) > 0.012:  # delete this for the original code
    logmel_spec = extract_logmel_spectrogram(audio)
    save_spectrogram(logmel_spec, time_of_event, minute)


def predict_image(image_filename, model):
    top5 = []

    img = load_img(image_filename, target_size=(IMG_SIZE, IMG_SIZE))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    value = model.predict(x)
    value = value[0]

    for i2 in sorted(value, reverse=True)[:5]:
        idx = np.where(value == i2)
        idx = idx[0][0]
        percent = np.round(value[idx] * 100, 3)
        species_and_percent = f"{species[idx]} , {percent}%"
        top5.append(species_and_percent)
        species_dictionary[species[idx]] += percent
    print(f"Top-5 species at {image_filename}: {top5}")
    return species_dictionary


def create_csv(species_dictionary, filename, pd_index, pred_df):
    sorted_dictionary = []
    ## Sort the dictionary in descending order based on the percent values
    sorted_dictionary = sorted(species_dictionary.items(), key=lambda x: x[1], reverse=True)
    pd_dict = {}
    pd_dict["Recording_Name"] = filename

    for idx, i in enumerate(sorted_dictionary[:5]):
        p = np.round(sorted_dictionary[idx][1], 3)
        top_idx = f"Top{idx + 1}"
        topval_idx = f"Top{idx + 1} Value"
        pd_dict[top_idx] = i[0]
        pd_dict[topval_idx] = p

    if pd_index == 0:
        pred_df = pd.DataFrame(data=pd_dict, index=[pd_index])
        pd_index += 1
    else:
        temp_df = pd.DataFrame(data=pd_dict, index=[pd_index])
        pd_index += 1
        pred_df = pd.concat([pred_df, temp_df], ignore_index=False)
    return pd_index, pred_df


def event_detector(signal, sample, minute):
    number_of_buffers = len(signal) / buffer

    for current_buffer in range(1, int(number_of_buffers) + 1):
        event_counter = 0
        start = buffer * (current_buffer - 1)

        ## Getting the Relative Extrema of the signal array
        buffered_signal = signal[start:(buffer * current_buffer)]
        buffered_array = np.array(buffered_signal)
        extrema = argrelextrema(buffered_array, np.greater, order=250) # 100 originally

        ## For each element of the relative extrema, check if above the threshold of 0.25 for detection of an event
        for i in extrema[0]:
            local_max = buffered_signal[i]
            if local_max > EVENT_THRESHOLD and np.max(buffered_array) < 0.5:   # 0.5 value is to disregard extremely loud sounds such as barking of a dog near the microphone
                event_counter += 1
        ## If an event is detected, the event counter should not be equal to 0
        if event_counter != 0:
            time_of_event = current_buffer * seconds_per_buffer
            preprocess_and_save_spectrogram(sample, time_of_event, minute)
        elif event_counter == 0:
            continue

if __name__ == "__main__":
    global OFFSET # Set these variables to global para ma-access ng mga malalalim na functions/nested functions
    OFFSET = 0
    predicted_species_list = []
    species_dictionary = {}
    pred_df = {}
    pd_df = {}
    pd_index = 0
    for i in species:
        species_dictionary[i] = 0

    ## Loading the CNN Model ; This is done in the start to avoid loading the model repeatedly during inference.
    print("LOADING THE CNN MODEL")
    start_time = time.time()
    model = load_cnn_model(MODEL_PATH)
    loading_time = time.time()
    print(f'Time to load the model: {np.round(loading_time - start_time, 2)} seconds\n')

    for minute in range(1, 61):
        print(f"{minute - 1}:00 - {minute}:00")
        ## Loading the wavfile
        print("LOADING THE AUDIO FILE")
        start_time = time.time()
        OFFSET = 60 * (minute - 1)
        signal = load_wavfile(sample, OFFSET)
        end_time = time.time()
        print(f"Time to load the audio file: {np.round(end_time - start_time, 2)} seconds")

        # Preprocessing Stage
        print("PREPROCESSING STAGE")
        start_time = time.time()
        event_detector(signal, sample, minute)
        preprocess_time = time.time()
        print(f'Total Preprocessing Time: {np.round(preprocess_time - start_time, 2)} seconds\n')

    # Inference Stage DO THIS BY BATCH
    print("INFERENCE STAGE")
    start_time = time.time()
    os.chdir(WAVFILE_SPECTROGRAM)
    image_files = os.listdir()
    image_files = os_sorted(image_files)
    print(f"Total number of images: {len(image_files)}")
    for file in image_files:
        ## To reset the values per image
        for i in species:
            species_dictionary[i] = 0

        filename, ext = os.path.splitext(file)
        if ext == '.png':
            top5_species = predict_image(file, model)
            pd_index, pred_df = create_csv(top5_species, filename, pd_index, pred_df)

    pred_df.to_csv("predictions.csv", index=False)
    inference_time = time.time()
    print(f'Inference Time: {np.round(inference_time - start_time, 2)} seconds')
    print("END")