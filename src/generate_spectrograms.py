import os
import audioread
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import warnings

warnings.filterwarnings("ignore")

def generate_spectrograms(input_directory):
    supported_formats = ('.ogg', '.wav', '.webm')

    for filename in os.listdir(input_directory):
        if filename.endswith(supported_formats):
            file_path = os.path.join(input_directory, filename)
            
            try:
                with audioread.audio_open(file_path) as f:
                    sr = f.samplerate
                    y = np.zeros(0)
                    
                    for buf in f:
                        y = np.concatenate((y, np.frombuffer(buf, dtype=np.float32)))
                
                if not np.all(np.isfinite(y)):
                    y = y[np.isfinite(y)]
                    print(f"Removed non-finite values from {filename}")

                S = librosa.stft(y)
                S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

                # Create the spectrogram plot
                plt.figure(figsize=(10, 10), dpi=25.5)
                plt.axis('off')
                librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

                # Generate the output filename and save the plot
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(input_directory, output_filename)
                plt.savefig(output_path)
                plt.close()

                print(f"Generated spectrogram for {filename} -> {output_filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

audio_directory = '/mnt/d/Productivity/CSE715/cough-type-clustering/cough_dataset/'
generate_spectrograms(audio_directory)