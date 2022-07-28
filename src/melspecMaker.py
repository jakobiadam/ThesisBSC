#imports
import numpy as np
import glob
import h5py
import time
import librosa
import os
import argparse

#globals
n_fft = 1024
sr = 22050
hop_size = 315
n_mels = 80
f_mel_min = 50 #Hz
f_mel_max = 11000 #Hz
power = 2 #magnitude spektrogram
window = "hann"
center = True
pad_mode = "reflect"
ref = 1.0
amin = 1e-10
top_db = 80.0

def envelope(y,acc=1e-6,forward=False):
    """A függvény kap egy array-t, mint paraméter, forward = False esetén a legelső elemét eggyel,
       a legutolsó elemét pedig az acc paraméter értékével szorozza be (forward = True esetén fordítva),
       a köztes elemeket a két végpontot összekötő exponenciális burkológörbe értékeivel szorozza be,
       végül a módosított array-el tér vissza.
       
       Paraméterek:
       ###########################################################################################
       y         :    az array, amit 'le akarunk csengetni'.
       acc       :    a paraméter, hogy az utolsó array értékünket
                      mennyire nullához közeli számmal szorozzuk be.
                      Alapértelmezett értéke 1e-6.
       forward   :    True - az első elem acc, az utolsó 1,
                      False - az első elem 1 és az utolsó acc
    """
    x = np.arange(len(y))
    last = len(y) - 1
    a = np.log(acc) / last
    b = 0
    if forward:
        a = -a
        b = np.log(acc)
        
    return np.exp(a * x + b) * y


def loop(data,edges=220,looplength=999*315):
    """Ha a beadott array megfelelő vagy több elemszámú, akkor a függvény változtatás nélkül visszaadja
       a beadott array első looplength elemét, egyébként a beadott arrayt a megfelelő hosszra loopolja úgy,
       hogy az exponenciális burkológörbét csak szükség esetén használja (vagyis a
       felvétel elején semmiképpen sem, a legvégén pedig csak fölfutásra, ha épp úgy jön ki).
       
       Paraméterek:
       ########################################################################
       data          :    a loopolni kívánt array
       edges         :    loopolásnál hány elemet szorozzon az exponenciális burkolóval
                          (külön az elejére és a végére)
       looplength    :    a bővített array kívánt hossza
       """
    
    midCount = int(np.ceil(looplength / len(data)) - 2) # A középső részarrayek száma, ha kisebb mint 0, akkor az array legalább a kívánt hosszúságú
    mid = 0
    if midCount < 0:
        return data[:looplength]
    elif midCount:
        mid = np.concatenate((envelope(data[:edges],forward=True),data[edges:-edges],envelope(data[-edges:])))
    
    first = np.concatenate((data[:-edges],envelope(data[-edges:])))
    last = np.concatenate((envelope(data[:edges],forward=True),data[edges:]))
    
    arr_list = [first]
    for i in range(midCount):
        arr_list.append(mid)
    arr_list.append(last)
    arr_new = np.concatenate(arr_list)
    return arr_new[:looplength]


def create_specgrams(src,dest,files):
    """A függvény végigiterál az src mappában lévő fájlokon, mindegyik hangfájlt beolvassa, újramintavételezi
       és szükség szerint loopolja a kívánt hosszig. Elkészíti a decibel skálájú spektrogramokat
       és elmenti a dest mappában az eredeti fájlnév + .h5 néven.
       
       Paraméterek:
       ################################################################################
       dest     :    Cél mappa
       files    :    String lista a fájl elérési útvonalakról (pl. '../data/audio/56.wav')
       """
    start = time.time()
    for i in range(len(files)):
        #Beolvasás, sr: 44100 -> 22050 Hz, loopolás
        y,sr = librosa.core.load(files[i], sr = None) #dtype float32
        if (sr != 44100):
            print("Nem volt jó az alapfeltevésem, van olyan felvétel, aminek a mintavételezési frekvenciája nem egyenlő 44100-al.")
            return
        sr = sr // 2
        y = y[::2] # gyors változat: 44100 Hz -> 22050 Hz sr
        y = loop(y) #loopolás (szükség esetén loopol)
        
        #Spektrogram elkészítése
        S = librosa.feature.melspectrogram(y,sr,n_fft=n_fft,hop_length=hop_size,win_length=n_fft,window=window,
                                       center=True,pad_mode=pad_mode,power=power,n_mels=n_mels,
                                       fmin=f_mel_min,fmax=f_mel_max)
        
        #Decibel skála
        #S = librosa.core.power_to_db(S, ref=ref, amin=amin, top_db=top_db)
        
        #Az eredeti megoldásban sima logaritmus szerepelt
        eps = 2.220446049250313e-16
        np.maximum(S,eps,S)
        np.log(S,S)
        S = np.array(S, dtype=np.float32)
        
        #Fájlba írás hdf5 tömörítéssel
        f = h5py.File(f"{dest}/{files[i][len(src):]}.h5", 'w')
        dset = f.create_dataset("S", S.shape, dtype=S.dtype)
        dset.write_direct(S)
        f.close()
        
        #Futás állapotának kiírása
        if (i % (len(files) // 20) == 0):
            os.system("clear")
            print(f"A futás állása: {i/len(files)*100} %.")
            print(f"Az eddig eltelt idő: {time.time() - start} s.")
    print("Futás vége! Fájlok a dest mappában!")
    
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir_path", help="Forrás mappa, mely tartalmazza az átalakítani kívánt .wav fájlokat.")
    parser.add_argument("dest_dir_path", help="Ebbe a mappába lesznek elmentve az elkészült spektrogramok.")
    args = parser.parse_args()
    
    src_dir = args.src_dir_path
    dest_dir = args.dest_dir_path
    files = glob.glob(src_dir + "/*.wav")
    
    create_specgrams(src_dir,dest_dir,files)
    
    print("Program vége")
    return 0

if __name__ == "__main__":
    main()

