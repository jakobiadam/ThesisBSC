#Imports
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import h5py
import math

import numpy as np
import random
import time

#from IPython.display import clear_output
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Hyperparameters
root = "."
cv_path = root + "/5FCV_labels"
spect_ff10_path = root + "/data/ff1010bird"
spect_warb_path = root + "/data/warblrb10k"
#fold = 0 #hányadik Cross Validation az 5-ből (0,1,2,3,4)
cnvmult=16 #channelek száma

max_cycle = 3 #ennyiszer fut le a .fit() függvény, itt muszáj 3-nak lennie!!!!!
epochsize = 1500 #egy .fit() függvényben az epochok száma
lr = 0.001 #learning rate (kezdeti) !!!!!!FÜGGVÉNYBEN IS ÉRTÉKADOK!!!!!

##################################################################################################################################################
def XFCV_read(cv_path,folders,id_paths,num):
    '''Beolvassa a szétbontott felvétel id-kat, és visszadja (x : id és y : label(int)) tuple-ként'''
    x = []
    y = []
    for i in range(len(folders)):
        f = open(f"{cv_path}/{folders[i]}/{num}.csv","r")
        string = f.read()
        arr = string.split('\n')
        for s in arr:
            split = s.split(",")
            x.append(f"{id_paths[i]}/{split[0]}.wav.h5")
            y.append(int(split[1]))

    random_index = [i for i in range(len(x))]
    random.shuffle(random_index)
    x = np.array(x)[random_index]
    y = np.array(y)[random_index]    
    return (x,y)
##################################################################################################################################################

#Az augmentáló custom layerem, ciklikusan rotál az időtengelyen és eltol véletlenszerűen +/- 1 melt a meltengelyen
##################################################################################################################################################
class BADAugment(tf.keras.layers.Layer):

    def __init__(self, **kwargs):    
        super(BADAugment, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BADAugment, self).build(input_shape)

    def call(self, input_data):

        # Ciklikus időbeli eltolás
        t_shift = tf.random.uniform(shape=[], minval=0, maxval=input_data.shape[2], dtype= tf.dtypes.int32)
        input_data = tf.roll(input_data,shift=t_shift,axis=2)

        #+/-1 mel eltolás
        # padding az interpoláláshoz, hogy ne csússzon ki a tartományból
        paddings = tf.constant([[0,0],[1,1], [0,0]])
        padded = tf.pad(input_data, paddings, "CONSTANT",constant_values=-20) # minél kisebb szám, annál kevésbé volt ott hang

        # interpoláció
        mel_shift = tf.random.uniform(shape=[], minval=-1, maxval=1, dtype=tf.dtypes.float32) # -1 és 1 közötti eltolás
        x_input = tf.linspace(start=0., stop=float(input_data.shape[1])-1.0, num=input_data.shape[1])
        input_data = tfp.math.interp_regular_1d_grid(x=x_input,x_ref_min=mel_shift-1,x_ref_max=float(input_data.shape[1])+mel_shift,y_ref=padded,axis=1)

        return input_data
##################################################################################################################################################

#Elemenkénti zérus átlagolás a mel tengelyre (az idő tengelyen keresztül). Custom layer
##################################################################################################################################################
class BADZeroMean(tf.keras.layers.Layer):

    def __init__(self, **kwargs):    
        super(BADZeroMean, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BADZeroMean, self).build(input_shape)

    def call(self, input_data):
        mean_tensor = tf.tile(tf.expand_dims(tf.reduce_mean(input_data,axis=2),
                              axis=-1),[1,1,input_data.shape[2]])

        #Ez az utolsó preprocessing layer, ezért itt létre kell hozni a channel dimenziót ahhoz,
        #hogy a konvulúciós layerek fogadni tudják a spektrogramokat.

        return tf.expand_dims(input_data - mean_tensor, axis=-1)
##################################################################################################################################################


def get_train_val_vars(fold):
    # A fenti listák tartalmazzák a split listákat.
    spects = []
    labels = []
    for i in range(1,6):
        sp, la = XFCV_read(cv_path,["ff1010bird","warblrb10k"],[spect_ff10_path,spect_warb_path],i)
        spects.append(sp)
        labels.append(la)

    #Ezek már a megfelelő indexhez tartozó train - validation - splitek, 
    #de a spect-ek még csak az id-ket tartalmazzák, a tényleges spektrogramokat még be kell olvasni.
    train_spects_ids = np.concatenate(spects[:fold]+spects[fold+1:],axis=-1)
    train_labels = np.concatenate(labels[:fold]+labels[fold+1:],axis=-1)
    val_spects_ids = spects[fold]
    val_labels = labels[fold]

    del spects
    del labels

    #Spektrogramok beolvasása
    start = time.time()
    train_spects = np.ones((len(train_spects_ids),80,1000),dtype=np.float32)
    for i in range(len(train_spects_ids)):
        f = h5py.File(train_spects_ids[i],'r')
        train_spects[i] = np.array(f[list(f.keys())[0]],dtype=np.float32)
        f.close()
        if i%50==0:
            print(f"ciklus:{i}, idő:{time.time()-start}")

    del train_spects_ids

    val_spects = np.ones((len(val_spects_ids),80,1000),dtype=np.float32)
    for i in range(len(val_spects_ids)):
        f = h5py.File(val_spects_ids[i],'r')
        val_spects[i] = np.array(f[list(f.keys())[0]],dtype=np.float32)
        f.close()
        if i%50==0:
            print(f"ciklus:{i}, idő:{time.time()-start}")

    del val_spects_ids
    return (train_spects, train_labels, val_spects, val_labels)

def learning_i(fold):

    train_spects, train_labels, val_spects, val_labels = get_train_val_vars(fold)

    #Model létrehozása
    ##################################################################################################################################################
    model = tf.keras.models.Sequential()

    #preprocessing
    model.add(BADAugment(input_shape=(80,1000),name="Augment"))
    model.add(tf.keras.layers.BatchNormalization(axis=1,name="BatchNorm",scale=False, center=False))
    model.add(BADZeroMean(name="ZeroMean"))

    #Convolution and pooling
    model.add(Conv2D(cnvmult, (3, 3)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(cnvmult, (3, 3)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(cnvmult, (1, 3))) #Az én spektrogramjaim alakja az eredeti megoldás spektrogramjainak alakjának
    model.add(LeakyReLU(alpha=0.01))   #a transzponáltja, ehhez igazítottam a maxpool kernel alakját is.
    model.add(MaxPooling2D(pool_size=(1, 3)))

    model.add(Conv2D(cnvmult, (1, 3)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(1, 3)))

    #Dense
    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    ##################################################################################################################################################

    #Tanítás és mentés
    lr = 0.0001
    loss = tf.keras.losses.BinaryCrossentropy()
    metric = tf.keras.metrics.AUC(num_thresholds=200) #itt talán lehetne 1000 is, az eredeti a 200.

    history = np.array([])

    for i in range(max_cycle):
        os.system("clear")
        #clear_output()
        print(f"Validation : {z+1} / 5")
        print(f"Fit ciklus : {i+1} / {max_cycle}")
        lr = lr / 10
        model.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics=metric) #compile
        history_i = model.fit(x=train_spects, y=train_labels, epochs=epochsize, callbacks=None,
                            validation_data=(val_spects,val_labels),shuffle=True)

        model.save(root + f'/models/model{fold+1}')

        if i == 0:
            history = np.array([history_i.history[list(history_i.history.keys())[i]] 
                          for i in range(len(list(history_i.history.keys())))], dtype=np.float32)
        else:
            history = np.concatenate((history,np.array([history_i.history[list(history_i.history.keys())[i]] 
                      for i in range(len(list(history_i.history.keys())))], dtype=np.float32)),axis=-1)
      
        np.savetxt(root+f'/models/model{fold+1}/history.csv',history.T,delimiter=',',header='loss,acc,val_loss,val_acc')

for z in range(5):
    learning_i(z)
