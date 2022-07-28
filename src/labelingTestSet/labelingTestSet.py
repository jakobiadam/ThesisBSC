#imports
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import subprocess
import librosa
import librosa.display
import h5py
import seaborn as sns
from IPython.display import Audio


#style
sns.set_theme()
dic = {'xtick.labelsize' : 15, 'ytick.labelsize' : 15}
plt.style.use(dic)


# konstansok
filePathSpec = '../../data/master/spect_2021/testdata/{}.wav.h5'
filePathAudio = '../../data/audio/testdata/{}.wav'
recordIDs = np.genfromtxt("testdataID_list.csv",dtype=str)

def count_rows():
    f = open("sub_testdata.csv",'r')
    s = f.read()
    f.close()
    l = s.split("\n")
    return len(l)

def specBeolvas(filePath,ID):
    file = filePath.format(ID)
    f = h5py.File(file)
    spec = np.array(f[list(f.keys())[0]])
    f.close()
    return spec

def plotSpec(spec):
    
    fig, ax = plt.subplots(figsize=(30,6))
    img = librosa.display.specshow(spec, x_axis='time',hop_length=315,
                             y_axis='mel', sr=22050,
                             fmin=50,fmax=11000, ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel-frequency spectrogram',fontsize=40)
    ax.set_xlabel("Time [s]",fontsize=30)
    ax.set_ylabel("Frequency [Hz]",fontsize=30)
    plt.show()

def main():
    for i in range(count_rows(),len(recordIDs)):
        spec = specBeolvas(filePathSpec,recordIDs[i])
        plotSpec(spec)
        
        cycle_test = 1
        while cycle_test:
            audioPath = filePathAudio.format(recordIDs[i])
           # subprocess.call(["play", audioPath])
            Audio(audioPath,rate=22050,autoplay=True)
            s = input("0 : No bird\n1 : Bird\nf : Finish\nelse : repeat record\n")
            if s == '0' or s == '1':
                f = open("sub_testdata.csv", "a")
                f.write(f"{recordIDs[i]},{s}\n")
                f.close()
                cycle_test = 0
            elif s == 'f':
                print(f"Program ended at index {i} / {len(recordIDs)}.")
                return 0
            else:
                continue
                
if __name__=="__main__":
    main()
