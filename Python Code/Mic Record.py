import os
import pyaudio
import wave
from datetime import date, datetime

## GLOBAL PARAMETERS
FRAMES_PER_BUFFER = 4410   ## Block of data na ipaprocess. e.g. if chunk = 22050 and rate == 44100, 0.5 seconds or 22050 chunk is processed and output then the next chunk
FORMAT = pyaudio.paInt16
CHANNELS = 1
# SR = 44100
SR = 22050
SECONDS = 3
TOTAL_FRAME_LENGTH = SECONDS * (SR / FRAMES_PER_BUFFER) # This is the total frame length of the n seconds. E.g. chunk = 22050, rate = 44100, and 10 seconds of recording, the total frame length is 20. 1 frame is 22050 sample points.

## GLOBAL PARAMETERS AND FILE PATHS #
MAIN_FOLDER = os.getcwd()
IMG_SIZE = 224
FRAME_SIZE = 2048
HOP_SIZE = 512
SR = 22050
N_MELS = 128
DURATION = 5.0
LOWCUT = 200
HIGHCUT = 11000


def record_to_file(file_path, previous_data, stream, p):
    frame_wav = []
    duration = 60 * 60 # 60 minutes
    while True:
        data = stream.read(FRAMES_PER_BUFFER)
        frame_wav.append(data)
        if len(frame_wav) == duration * (SR / FRAMES_PER_BUFFER):
            break
    frame_wav = previous_data + frame_wav   # DAHIL IN-ADD ANG DALAWANG FRAMES, 2 + 20 SECONDS OF FRAMES ANG OUTPUT. OUTPUT AUDIO IS 22 SECONDS LONG
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    sample_width = p.get_sample_size(FORMAT)
    wf.setsampwidth(sample_width)
    wf.setframerate(SR)
    wf.writeframes(b''.join(frame_wav))
    wf.close()


if __name__ == '__main__':
    date_today = date.today()
    t = datetime.now()
    t = t.strftime("%H")
    if t < "12":
        session = "BeforeDawn"
    elif t >= "12" and t < "17":
        session = "Lunchtime"
    elif t >= "17":
        session = "BeforeDusk"

    ## while True statement was used to loop the recording after the last stage of inference
    while True:
        # Calling pyadio module and starting recording
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SR,
                        input=True,
                        output=True,
                        frames_per_buffer=FRAMES_PER_BUFFER)

        stream.start_stream()
        print("Starting!")

        ## Recording data until under threshold
        frames = []
        frame_output = []
        predicted_species_list = []
        recording_number = 1
        output_filename = f"{date_today}_{session}_Recording{recording_number}.wav"

        ## To check the recording number. If number is used, increase by 1
        while True:
            if not os.path.exists(output_filename):
                break
            else:
                recording_number += 1
                output_filename = f"{date_today}_{session}_Recording{recording_number}.wav"

        record_to_file(output_filename, frame_output, stream, p)  # DI AKO MARUNONG NG struct.pack PARA IBALIK YUNG data_int PARA IPASOK SA RECORD FUNCTION
        print("Recording Finished")
        break