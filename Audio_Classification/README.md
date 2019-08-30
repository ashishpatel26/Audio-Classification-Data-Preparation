
# ESC-50 Audio Classification Using CNN

## Load the zip file and unzip and before check the GPU

#### This file is Run on Kaggle GPU or Colaboratory GPU

**Dataset Download :** https://github.com/karoldvl/ESC-50/archive/master.zip


```python
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

    Found GPU at: /device:GPU:0



```python
# import dependencies
import requests, zipfile, io
from glob import glob
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
seed = 7
import pandas as pd
np.random.seed(seed)
import os
```


```python
zip_file_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip' # link: ESC-50 Datset
```


```python
if not os.path.exists('sound'):
    os.makedirs('sound')
```


```python
r = requests.get(zip_file_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('sound/')
z.close()
```


```python
# glob('sound/ESC-50-master/audio/*')
```

# Define a function to covert the image based on calculate log scaled mel-spectrograms and their corresponding deltas from a sound clip.

Regarding fixed size input, we will divide each sound clip into segments of 60x41 (60 rows and 41 columns). The mel-spec and their deltas will become two channels, which we will be fed into CNN


```python
# !pip install librosa
import librosa
```


```python
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)

def extract_features(bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for fn in tqdm(glob('sound/ESC-50-master/audio/*')):
        sound_clip,s = librosa.load(fn) # 5sec
        sound_clip   = np.concatenate((sound_clip,sound_clip),axis=None) # make it 10s
        label = fn.split("/")[-1].split("-")[-1].split(".")[0]
        for (start,end) in windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.core.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                labels.append(label)
            
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    return np.array(features), np.array(labels,dtype = np.int)
```


```python
features,labels = extract_features()
```

    100%|██████████| 2000/2000 [14:03<00:00,  2.35it/s]
    /opt/conda/lib/python3.6/site-packages/scipy/signal/_arraytools.py:45: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      b = a[a_slice]



```python
# label category names
df = pd.read_csv(glob('sound/ESC-50-master/meta/esc50.csv')[0])
df = df[['target','category']]
df = df.drop_duplicates().reset_index(drop=True)
df = df.sort_values(by=['target']).reset_index(drop=True)
df.head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>dog</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>rooster</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>frog</td>
    </tr>
  </tbody>
</table>
</div>




```python
my_dict = {}
for i in range(len(df)):
    my_dict[df['target'][i]] = df['category'][i]
my_dict
```




    {0: 'dog',
     1: 'rooster',
     2: 'pig',
     3: 'cow',
     4: 'frog',
     5: 'cat',
     6: 'hen',
     7: 'insects',
     8: 'sheep',
     9: 'crow',
     10: 'rain',
     11: 'sea_waves',
     12: 'crackling_fire',
     13: 'crickets',
     14: 'chirping_birds',
     15: 'water_drops',
     16: 'wind',
     17: 'pouring_water',
     18: 'toilet_flush',
     19: 'thunderstorm',
     20: 'crying_baby',
     21: 'sneezing',
     22: 'clapping',
     23: 'breathing',
     24: 'coughing',
     25: 'footsteps',
     26: 'laughing',
     27: 'brushing_teeth',
     28: 'snoring',
     29: 'drinking_sipping',
     30: 'door_wood_knock',
     31: 'mouse_click',
     32: 'keyboard_typing',
     33: 'door_wood_creaks',
     34: 'can_opening',
     35: 'washing_machine',
     36: 'vacuum_cleaner',
     37: 'clock_alarm',
     38: 'clock_tick',
     39: 'glass_breaking',
     40: 'helicopter',
     41: 'chainsaw',
     42: 'siren',
     43: 'car_horn',
     44: 'engine',
     45: 'train',
     46: 'church_bells',
     47: 'airplane',
     48: 'fireworks',
     49: 'hand_saw'}




```python
seed = 4
rng = np.random.RandomState(seed)
from keras.utils import to_categorical
```

    Using TensorFlow backend.



```python
onehot_labels = to_categorical(labels,num_classes=50)
```


```python
# Create train test Dataset

rnd_indices = np.random.rand(len(labels)) < 0.70

X_train = features[rnd_indices]
y_train = onehot_labels[rnd_indices]
X_test  = features[~rnd_indices]
y_test  = onehot_labels[~rnd_indices]
```


```python
X_train.shape, y_train.shape, X_test.shape, y_test.shape
```




    ((27998, 60, 41, 2), (27998, 50), (12002, 60, 41, 2), (12002, 50))



# CNN Model


```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten,InputLayer
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint
```


```python
def basemodel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(60,41,2), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='softmax'))
    # Compile model
    epochs = 25
    lrate = 0.01
    decay = lrate/epochs
#     sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
    return model

```


```python
if not os.path.exists('model'):
    os.makedirs('model')
    
filepath="model/weights_0.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
```


```python
model = basemodel()
print(model.summary())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_13 (Conv2D)           (None, 60, 41, 32)        608       
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 60, 41, 32)        0         
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, 60, 41, 32)        9248      
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 30, 20, 32)        0         
    _________________________________________________________________
    conv2d_15 (Conv2D)           (None, 30, 20, 64)        18496     
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 30, 20, 64)        0         
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, 30, 20, 64)        36928     
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 15, 10, 64)        0         
    _________________________________________________________________
    conv2d_17 (Conv2D)           (None, 15, 10, 128)       73856     
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 15, 10, 128)       0         
    _________________________________________________________________
    conv2d_18 (Conv2D)           (None, 15, 10, 128)       147584    
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 7, 5, 128)         0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 4480)              0         
    _________________________________________________________________
    dropout_16 (Dropout)         (None, 4480)              0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 1024)              4588544   
    _________________________________________________________________
    dropout_17 (Dropout)         (None, 1024)              0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 512)               524800    
    _________________________________________________________________
    dropout_18 (Dropout)         (None, 512)               0         
    _________________________________________________________________
    dense_9 (Dense)              (None, 50)                25650     
    =================================================================
    Total params: 5,425,714
    Trainable params: 5,425,714
    Non-trainable params: 0
    _________________________________________________________________
    None


# Training with Data Augmentation

One of the major reasons for overfitting is that we don’t have enough data to train our network. Apart from regularization, another very effective way to counter Overfitting is Data Augmentation. It is the process of artificially creating more images from the images you already have by changing the size, orientation etc of the image. It can be a tedious task but fortunately, this can be done in Keras using the ImageDataGenerator instance.


```python
from keras.preprocessing.image import ImageDataGenerator
```


```python
datagen = ImageDataGenerator(
              width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
              height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
              horizontal_flip=True,  # randomly flip images
              vertical_flip=False  # randomly flip images
          )
```


```python
# init the batch size and epochs

'''
Note: Due to Memory Error like Buffered data was truncated after reaching the output size limit. What i did is that Save the model in for example 60th epoch and close current program and run new program and restore saved model and train model from 61 epoch to 120 epoch and 
save that and close program and repeat this work for your interested epoch For this [100,50] three times repeat 

'''
batch_size = 50
epochs = 100
```


```python
# fit the model
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                              steps_per_epoch=int(np.ceil(X_train.shape[0] / float(batch_size))),
                              epochs=epochs,
                              validation_data=(X_test, y_test),
                              verbose=1,callbacks=callbacks_list)
 
```

    /opt/conda/lib/python3.6/site-packages/keras_preprocessing/image/numpy_array_iterator.py:127: UserWarning: NumpyArrayIterator is set to use the data format convention "channels_last" (channels on axis 3), i.e. expected either 1, 3, or 4 channels on axis 3. However, it was passed an array with shape (27998, 60, 41, 2) (2 channels).
      str(self.x.shape[channels_axis]) + ' channels).')


    Epoch 1/100
    560/560 [==============================] - 24s 42ms/step - loss: 3.5674 - acc: 0.0626 - val_loss: 3.2620 - val_acc: 0.1219
    
    Epoch 00001: val_acc improved from -inf to 0.12190, saving model to model/weights_0.best.hdf5
    Epoch 2/100
    560/560 [==============================] - 23s 40ms/step - loss: 3.0286 - acc: 0.1574 - val_loss: 2.7793 - val_acc: 0.2208
    
    Epoch 00002: val_acc improved from 0.12190 to 0.22080, saving model to model/weights_0.best.hdf5
    Epoch 3/100
    560/560 [==============================] - 23s 40ms/step - loss: 2.6666 - acc: 0.2366 - val_loss: 2.5812 - val_acc: 0.2770
    
    Epoch 00003: val_acc improved from 0.22080 to 0.27695, saving model to model/weights_0.best.hdf5
    Epoch 4/100
    560/560 [==============================] - 22s 40ms/step - loss: 2.3895 - acc: 0.3063 - val_loss: 2.2632 - val_acc: 0.3437
    
    Epoch 00004: val_acc improved from 0.27695 to 0.34369, saving model to model/weights_0.best.hdf5
    Epoch 5/100
    560/560 [==============================] - 23s 40ms/step - loss: 2.2095 - acc: 0.3525 - val_loss: 2.0672 - val_acc: 0.3970
    
    Epoch 00005: val_acc improved from 0.34369 to 0.39702, saving model to model/weights_0.best.hdf5
    Epoch 6/100
    560/560 [==============================] - 22s 40ms/step - loss: 2.0621 - acc: 0.3904 - val_loss: 2.0769 - val_acc: 0.3941
    
    Epoch 00006: val_acc did not improve from 0.39702
    Epoch 7/100
    560/560 [==============================] - 23s 40ms/step - loss: 1.9408 - acc: 0.4253 - val_loss: 2.0249 - val_acc: 0.4121
    
    Epoch 00007: val_acc improved from 0.39702 to 0.41210, saving model to model/weights_0.best.hdf5
    Epoch 8/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.8569 - acc: 0.4449 - val_loss: 1.8749 - val_acc: 0.4443
    
    Epoch 00008: val_acc improved from 0.41210 to 0.44426, saving model to model/weights_0.best.hdf5
    Epoch 9/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.7641 - acc: 0.4699 - val_loss: 1.9020 - val_acc: 0.4478
    
    Epoch 00009: val_acc improved from 0.44426 to 0.44784, saving model to model/weights_0.best.hdf5
    Epoch 10/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.6993 - acc: 0.4903 - val_loss: 1.7551 - val_acc: 0.4779
    
    Epoch 00010: val_acc improved from 0.44784 to 0.47792, saving model to model/weights_0.best.hdf5
    Epoch 11/100
    560/560 [==============================] - 23s 41ms/step - loss: 1.6302 - acc: 0.5084 - val_loss: 1.8264 - val_acc: 0.4668
    
    Epoch 00011: val_acc did not improve from 0.47792
    Epoch 12/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.5584 - acc: 0.5278 - val_loss: 1.6872 - val_acc: 0.5005
    
    Epoch 00012: val_acc improved from 0.47792 to 0.50050, saving model to model/weights_0.best.hdf5
    Epoch 13/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.5139 - acc: 0.5377 - val_loss: 1.6516 - val_acc: 0.5117
    
    Epoch 00013: val_acc improved from 0.50050 to 0.51175, saving model to model/weights_0.best.hdf5
    Epoch 14/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.4554 - acc: 0.5545 - val_loss: 1.7127 - val_acc: 0.4969
    
    Epoch 00014: val_acc did not improve from 0.51175
    Epoch 15/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.4088 - acc: 0.5718 - val_loss: 1.7446 - val_acc: 0.4958
    
    Epoch 00015: val_acc did not improve from 0.51175
    Epoch 16/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.3586 - acc: 0.5829 - val_loss: 1.6002 - val_acc: 0.5297
    
    Epoch 00016: val_acc improved from 0.51175 to 0.52975, saving model to model/weights_0.best.hdf5
    Epoch 17/100
    560/560 [==============================] - 23s 40ms/step - loss: 1.3279 - acc: 0.5944 - val_loss: 1.5813 - val_acc: 0.5336
    
    Epoch 00017: val_acc improved from 0.52975 to 0.53358, saving model to model/weights_0.best.hdf5
    Epoch 18/100
    560/560 [==============================] - 23s 40ms/step - loss: 1.2782 - acc: 0.6064 - val_loss: 1.6355 - val_acc: 0.5412
    
    Epoch 00018: val_acc improved from 0.53358 to 0.54116, saving model to model/weights_0.best.hdf5
    Epoch 19/100
    560/560 [==============================] - 23s 40ms/step - loss: 1.2341 - acc: 0.6174 - val_loss: 1.5320 - val_acc: 0.5470
    
    Epoch 00019: val_acc improved from 0.54116 to 0.54699, saving model to model/weights_0.best.hdf5
    Epoch 20/100
    560/560 [==============================] - 23s 41ms/step - loss: 1.2071 - acc: 0.6288 - val_loss: 1.4417 - val_acc: 0.5770
    
    Epoch 00020: val_acc improved from 0.54699 to 0.57699, saving model to model/weights_0.best.hdf5
    Epoch 21/100
    560/560 [==============================] - 23s 41ms/step - loss: 1.1796 - acc: 0.6349 - val_loss: 1.3945 - val_acc: 0.5851
    
    Epoch 00021: val_acc improved from 0.57699 to 0.58507, saving model to model/weights_0.best.hdf5
    Epoch 22/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.1467 - acc: 0.6473 - val_loss: 1.4604 - val_acc: 0.5797
    
    Epoch 00022: val_acc did not improve from 0.58507
    Epoch 23/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.1149 - acc: 0.6554 - val_loss: 1.6459 - val_acc: 0.5420
    
    Epoch 00023: val_acc did not improve from 0.58507
    Epoch 24/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.0948 - acc: 0.6596 - val_loss: 1.3876 - val_acc: 0.5964
    
    Epoch 00024: val_acc improved from 0.58507 to 0.59640, saving model to model/weights_0.best.hdf5
    Epoch 25/100
    560/560 [==============================] - 23s 41ms/step - loss: 1.0685 - acc: 0.6682 - val_loss: 1.4177 - val_acc: 0.5914
    
    Epoch 00025: val_acc did not improve from 0.59640
    Epoch 26/100
    560/560 [==============================] - 22s 40ms/step - loss: 1.0456 - acc: 0.6727 - val_loss: 1.4539 - val_acc: 0.5897
    
    Epoch 00026: val_acc did not improve from 0.59640
    Epoch 27/100
    560/560 [==============================] - 23s 40ms/step - loss: 1.0139 - acc: 0.6836 - val_loss: 1.3737 - val_acc: 0.6050
    
    Epoch 00027: val_acc improved from 0.59640 to 0.60498, saving model to model/weights_0.best.hdf5
    Epoch 28/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.9975 - acc: 0.6911 - val_loss: 1.3511 - val_acc: 0.6144
    
    Epoch 00028: val_acc improved from 0.60498 to 0.61440, saving model to model/weights_0.best.hdf5
    Epoch 29/100
    560/560 [==============================] - 23s 40ms/step - loss: 0.9733 - acc: 0.6965 - val_loss: 1.4043 - val_acc: 0.6043
    
    Epoch 00029: val_acc did not improve from 0.61440
    Epoch 30/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.9624 - acc: 0.7008 - val_loss: 1.3815 - val_acc: 0.6071
    
    Epoch 00030: val_acc did not improve from 0.61440
    Epoch 31/100
    560/560 [==============================] - 23s 40ms/step - loss: 0.9431 - acc: 0.7062 - val_loss: 1.2999 - val_acc: 0.6215
    
    Epoch 00031: val_acc improved from 0.61440 to 0.62148, saving model to model/weights_0.best.hdf5
    Epoch 32/100
    560/560 [==============================] - 23s 40ms/step - loss: 0.9382 - acc: 0.7070 - val_loss: 1.3587 - val_acc: 0.6199
    
    Epoch 00032: val_acc did not improve from 0.62148
    Epoch 33/100
    560/560 [==============================] - 23s 41ms/step - loss: 0.9157 - acc: 0.7152 - val_loss: 1.2994 - val_acc: 0.6261
    
    Epoch 00033: val_acc improved from 0.62148 to 0.62606, saving model to model/weights_0.best.hdf5
    Epoch 34/100
    560/560 [==============================] - 23s 41ms/step - loss: 0.8975 - acc: 0.7221 - val_loss: 1.2496 - val_acc: 0.6440
    
    Epoch 00034: val_acc improved from 0.62606 to 0.64398, saving model to model/weights_0.best.hdf5
    Epoch 35/100
    560/560 [==============================] - 23s 40ms/step - loss: 0.8822 - acc: 0.7255 - val_loss: 1.3239 - val_acc: 0.6263
    
    Epoch 00035: val_acc did not improve from 0.64398
    Epoch 36/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.8616 - acc: 0.7332 - val_loss: 1.3159 - val_acc: 0.6298
    
    Epoch 00036: val_acc did not improve from 0.64398
    Epoch 37/100
    560/560 [==============================] - 23s 40ms/step - loss: 0.8535 - acc: 0.7328 - val_loss: 1.2138 - val_acc: 0.6538
    
    Epoch 00037: val_acc improved from 0.64398 to 0.65381, saving model to model/weights_0.best.hdf5
    Epoch 38/100
    560/560 [==============================] - 23s 41ms/step - loss: 0.8373 - acc: 0.7374 - val_loss: 1.3271 - val_acc: 0.6324
    
    Epoch 00038: val_acc did not improve from 0.65381
    Epoch 39/100
    560/560 [==============================] - 23s 40ms/step - loss: 0.8250 - acc: 0.7428 - val_loss: 1.3692 - val_acc: 0.6309
    
    Epoch 00039: val_acc did not improve from 0.65381
    Epoch 40/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.8150 - acc: 0.7447 - val_loss: 1.3393 - val_acc: 0.6363
    
    Epoch 00040: val_acc did not improve from 0.65381
    Epoch 41/100
    560/560 [==============================] - 23s 41ms/step - loss: 0.7919 - acc: 0.7521 - val_loss: 1.2517 - val_acc: 0.6531
    
    Epoch 00041: val_acc did not improve from 0.65381
    Epoch 42/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.7903 - acc: 0.7535 - val_loss: 1.2711 - val_acc: 0.6480
    
    Epoch 00042: val_acc did not improve from 0.65381
    Epoch 43/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.7793 - acc: 0.7569 - val_loss: 1.1919 - val_acc: 0.6661
    
    Epoch 00043: val_acc improved from 0.65381 to 0.66606, saving model to model/weights_0.best.hdf5
    Epoch 44/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.7664 - acc: 0.7583 - val_loss: 1.2112 - val_acc: 0.6642
    
    Epoch 00044: val_acc did not improve from 0.66606
    Epoch 45/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.7617 - acc: 0.7648 - val_loss: 1.2689 - val_acc: 0.6525
    
    Epoch 00045: val_acc did not improve from 0.66606
    Epoch 46/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.7534 - acc: 0.7638 - val_loss: 1.2992 - val_acc: 0.6502
    
    Epoch 00046: val_acc did not improve from 0.66606
    Epoch 47/100
    560/560 [==============================] - 23s 40ms/step - loss: 0.7446 - acc: 0.7663 - val_loss: 1.1896 - val_acc: 0.6637
    
    Epoch 00047: val_acc did not improve from 0.66606
    Epoch 48/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.7315 - acc: 0.7685 - val_loss: 1.2755 - val_acc: 0.6562
    
    Epoch 00048: val_acc did not improve from 0.66606
    Epoch 49/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.7236 - acc: 0.7724 - val_loss: 1.1707 - val_acc: 0.6716
    
    Epoch 00049: val_acc improved from 0.66606 to 0.67164, saving model to model/weights_0.best.hdf5
    Epoch 50/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.7105 - acc: 0.7750 - val_loss: 1.2181 - val_acc: 0.6683
    
    Epoch 00050: val_acc did not improve from 0.67164
    Epoch 51/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.7111 - acc: 0.7777 - val_loss: 1.2469 - val_acc: 0.6646
    
    Epoch 00051: val_acc did not improve from 0.67164
    Epoch 52/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.6954 - acc: 0.7818 - val_loss: 1.2281 - val_acc: 0.6677
    
    Epoch 00052: val_acc did not improve from 0.67164
    Epoch 53/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.6902 - acc: 0.7839 - val_loss: 1.2124 - val_acc: 0.6765
    
    Epoch 00053: val_acc improved from 0.67164 to 0.67647, saving model to model/weights_0.best.hdf5
    Epoch 54/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.6858 - acc: 0.7869 - val_loss: 1.1980 - val_acc: 0.6756
    
    Epoch 00054: val_acc did not improve from 0.67647
    Epoch 55/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.6881 - acc: 0.7827 - val_loss: 1.1988 - val_acc: 0.6791
    
    Epoch 00055: val_acc improved from 0.67647 to 0.67914, saving model to model/weights_0.best.hdf5
    Epoch 56/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.6711 - acc: 0.7886 - val_loss: 1.1429 - val_acc: 0.6886
    
    Epoch 00056: val_acc improved from 0.67914 to 0.68864, saving model to model/weights_0.best.hdf5
    Epoch 57/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.6658 - acc: 0.7924 - val_loss: 1.1657 - val_acc: 0.6875
    
    Epoch 00057: val_acc did not improve from 0.68864
    Epoch 58/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.6656 - acc: 0.7888 - val_loss: 1.2439 - val_acc: 0.6686
    
    Epoch 00058: val_acc did not improve from 0.68864
    Epoch 59/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.6508 - acc: 0.7968 - val_loss: 1.2210 - val_acc: 0.6808
    
    Epoch 00059: val_acc did not improve from 0.68864
    Epoch 60/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.6507 - acc: 0.7965 - val_loss: 1.2337 - val_acc: 0.6726
    
    Epoch 00060: val_acc did not improve from 0.68864
    Epoch 61/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.6345 - acc: 0.8014 - val_loss: 1.2357 - val_acc: 0.6758
    
    Epoch 00061: val_acc did not improve from 0.68864
    Epoch 62/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.6310 - acc: 0.8032 - val_loss: 1.2572 - val_acc: 0.6737
    
    Epoch 00062: val_acc did not improve from 0.68864
    Epoch 63/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.6255 - acc: 0.8039 - val_loss: 1.2986 - val_acc: 0.6626
    
    Epoch 00063: val_acc did not improve from 0.68864
    Epoch 64/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.6262 - acc: 0.8035 - val_loss: 1.2321 - val_acc: 0.6753
    
    Epoch 00064: val_acc did not improve from 0.68864
    Epoch 65/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.6326 - acc: 0.8013 - val_loss: 1.1327 - val_acc: 0.6906
    
    Epoch 00065: val_acc improved from 0.68864 to 0.69063, saving model to model/weights_0.best.hdf5
    Epoch 66/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.6123 - acc: 0.8080 - val_loss: 1.2123 - val_acc: 0.6828
    
    Epoch 00066: val_acc did not improve from 0.69063
    Epoch 67/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.6087 - acc: 0.8101 - val_loss: 1.1973 - val_acc: 0.6864
    
    Epoch 00067: val_acc did not improve from 0.69063
    Epoch 68/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.6026 - acc: 0.8111 - val_loss: 1.1333 - val_acc: 0.6992
    
    Epoch 00068: val_acc improved from 0.69063 to 0.69922, saving model to model/weights_0.best.hdf5
    Epoch 69/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5930 - acc: 0.8156 - val_loss: 1.2433 - val_acc: 0.6766
    
    Epoch 00069: val_acc did not improve from 0.69922
    Epoch 70/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5980 - acc: 0.8118 - val_loss: 1.1892 - val_acc: 0.6911
    
    Epoch 00070: val_acc did not improve from 0.69922
    Epoch 71/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5991 - acc: 0.8112 - val_loss: 1.1768 - val_acc: 0.6909
    
    Epoch 00071: val_acc did not improve from 0.69922
    Epoch 72/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5879 - acc: 0.8159 - val_loss: 1.1697 - val_acc: 0.6937
    
    Epoch 00072: val_acc did not improve from 0.69922
    Epoch 73/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5855 - acc: 0.8188 - val_loss: 1.0784 - val_acc: 0.7099
    
    Epoch 00073: val_acc improved from 0.69922 to 0.70988, saving model to model/weights_0.best.hdf5
    Epoch 74/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5795 - acc: 0.8162 - val_loss: 1.1160 - val_acc: 0.7041
    
    Epoch 00074: val_acc did not improve from 0.70988
    Epoch 75/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5857 - acc: 0.8178 - val_loss: 1.2015 - val_acc: 0.6842
    
    Epoch 00075: val_acc did not improve from 0.70988
    Epoch 76/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5766 - acc: 0.8192 - val_loss: 1.1359 - val_acc: 0.7030
    
    Epoch 00076: val_acc did not improve from 0.70988
    Epoch 77/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5735 - acc: 0.8212 - val_loss: 1.1641 - val_acc: 0.6990
    
    Epoch 00077: val_acc did not improve from 0.70988
    Epoch 78/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5642 - acc: 0.8257 - val_loss: 1.1240 - val_acc: 0.7066
    
    Epoch 00078: val_acc did not improve from 0.70988
    Epoch 79/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5647 - acc: 0.8243 - val_loss: 1.1312 - val_acc: 0.7032
    
    Epoch 00079: val_acc did not improve from 0.70988
    Epoch 80/100
    560/560 [==============================] - 22s 40ms/step - loss: 0.5590 - acc: 0.8241 - val_loss: 1.1340 - val_acc: 0.7053
    
    Epoch 00080: val_acc did not improve from 0.70988
    Epoch 81/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5556 - acc: 0.8249 - val_loss: 1.1243 - val_acc: 0.7099
    
    Epoch 00081: val_acc improved from 0.70988 to 0.70988, saving model to model/weights_0.best.hdf5
    Epoch 82/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5521 - acc: 0.8284 - val_loss: 1.2078 - val_acc: 0.6921
    
    Epoch 00082: val_acc did not improve from 0.70988
    Epoch 83/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5516 - acc: 0.8278 - val_loss: 1.1587 - val_acc: 0.7036
    
    Epoch 00083: val_acc did not improve from 0.70988
    Epoch 84/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5477 - acc: 0.8291 - val_loss: 1.1231 - val_acc: 0.7086
    
    Epoch 00084: val_acc did not improve from 0.70988
    Epoch 85/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5486 - acc: 0.8297 - val_loss: 1.1407 - val_acc: 0.7041
    
    Epoch 00085: val_acc did not improve from 0.70988
    Epoch 86/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5385 - acc: 0.8333 - val_loss: 1.0560 - val_acc: 0.7211
    
    Epoch 00086: val_acc improved from 0.70988 to 0.72113, saving model to model/weights_0.best.hdf5
    Epoch 87/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5413 - acc: 0.8327 - val_loss: 1.1869 - val_acc: 0.6945
    
    Epoch 00087: val_acc did not improve from 0.72113
    Epoch 88/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5314 - acc: 0.8358 - val_loss: 1.1158 - val_acc: 0.7122
    
    Epoch 00088: val_acc did not improve from 0.72113
    Epoch 89/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5322 - acc: 0.8346 - val_loss: 1.0899 - val_acc: 0.7194
    
    Epoch 00089: val_acc did not improve from 0.72113
    Epoch 90/100
    560/560 [==============================] - 22s 38ms/step - loss: 0.5195 - acc: 0.8397 - val_loss: 1.1924 - val_acc: 0.7000
    
    Epoch 00090: val_acc did not improve from 0.72113
    Epoch 91/100
    560/560 [==============================] - 22s 38ms/step - loss: 0.5270 - acc: 0.8352 - val_loss: 1.1305 - val_acc: 0.7130
    
    Epoch 00091: val_acc did not improve from 0.72113
    Epoch 92/100
    560/560 [==============================] - 21s 38ms/step - loss: 0.5220 - acc: 0.8361 - val_loss: 1.1054 - val_acc: 0.7180
    
    Epoch 00092: val_acc did not improve from 0.72113
    Epoch 93/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5184 - acc: 0.8392 - val_loss: 1.1558 - val_acc: 0.7081
    
    Epoch 00093: val_acc did not improve from 0.72113
    Epoch 94/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5203 - acc: 0.8365 - val_loss: 1.1274 - val_acc: 0.7113
    
    Epoch 00094: val_acc did not improve from 0.72113
    Epoch 95/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5244 - acc: 0.8368 - val_loss: 1.1245 - val_acc: 0.7133
    
    Epoch 00095: val_acc did not improve from 0.72113
    Epoch 96/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5105 - acc: 0.8421 - val_loss: 1.1957 - val_acc: 0.7023
    
    Epoch 00096: val_acc did not improve from 0.72113
    Epoch 97/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5056 - acc: 0.8420 - val_loss: 1.1408 - val_acc: 0.7102
    
    Epoch 00097: val_acc did not improve from 0.72113
    Epoch 98/100
    560/560 [==============================] - 22s 39ms/step - loss: 0.5152 - acc: 0.8391 - val_loss: 1.1755 - val_acc: 0.7031
    
    Epoch 00098: val_acc did not improve from 0.72113
    Epoch 99/100
    560/560 [==============================] - 21s 38ms/step - loss: 0.5067 - acc: 0.8413 - val_loss: 1.2164 - val_acc: 0.6982
    
    Epoch 00099: val_acc did not improve from 0.72113
    Epoch 100/100
    560/560 [==============================] - 21s 38ms/step - loss: 0.5058 - acc: 0.8427 - val_loss: 1.1470 - val_acc: 0.7108
    
    Epoch 00100: val_acc did not improve from 0.72113


Note: Due to Memory Error like Buffered data was truncated after reaching the output size limit.
What i did is that Save the model in for example 60th epoch and close current program and run new program and restore saved model and train model from 61 epoch to 120 epoch and save that and close program and repeat this work for your interested epoch 
For this [100,50] three times repeat 



```python
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Plot training & validation accuracy values
plt.figure(figsize=(15,6))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(15,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

![png](https://github.com/sw-ot-ashishpatel/Audio-Classification-Data-Preparation/blob/master/Audio_Classification/Images/output_30_0.png)



![png](https://github.com/sw-ot-ashishpatel/Audio-Classification-Data-Preparation/blob/master/Audio_Classification/Images/output_30_1.png)



```python
# evaluate model
model.evaluate(X_test, y_test)
```

    12002/12002 [==============================] - 1s 111us/step





    [1.1470031869786914, 0.71079820029995]



# Classification Report and Confusion Matrix


```python
from sklearn.metrics import classification_report, confusion_matrix
```


```python
y_pred = model.predict_classes(X_test)
```


```python
target_name = np.array(df['category'])
```


```python
print(classification_report(np.argmax(y_test,axis=1),y_pred,target_names=target_name))
```

                      precision    recall  f1-score   support
    
                 dog       0.76      0.76      0.76       242
             rooster       0.76      0.42      0.54       248
                 pig       0.82      0.62      0.71       253
                 cow       0.74      0.69      0.71       233
                frog       0.92      0.86      0.89       237
                 cat       0.64      0.63      0.64       236
                 hen       0.72      0.68      0.70       219
             insects       0.73      0.79      0.76       230
               sheep       0.73      0.76      0.74       243
                crow       0.88      0.72      0.79       235
                rain       0.80      0.94      0.87       220
           sea_waves       0.64      0.88      0.74       242
      crackling_fire       0.88      0.77      0.82       247
            crickets       0.89      0.94      0.91       245
      chirping_birds       0.90      0.67      0.77       232
         water_drops       0.76      0.50      0.60       250
                wind       0.57      0.93      0.71       229
       pouring_water       0.84      0.63      0.72       234
        toilet_flush       0.76      0.78      0.77       244
        thunderstorm       0.70      0.79      0.74       241
         crying_baby       0.71      0.71      0.71       252
            sneezing       0.54      0.30      0.39       267
            clapping       0.88      0.81      0.84       219
           breathing       0.74      0.66      0.70       245
            coughing       0.68      0.41      0.51       271
           footsteps       0.79      0.80      0.80       239
            laughing       0.69      0.53      0.60       244
      brushing_teeth       0.88      0.86      0.87       250
             snoring       0.59      0.72      0.65       254
    drinking_sipping       0.63      0.47      0.54       247
     door_wood_knock       0.59      0.59      0.59       233
         mouse_click       0.71      0.60      0.65       225
     keyboard_typing       0.84      0.77      0.80       245
    door_wood_creaks       0.70      0.58      0.63       237
         can_opening       0.69      0.42      0.52       261
     washing_machine       0.79      0.63      0.70       228
      vacuum_cleaner       0.79      0.93      0.85       230
         clock_alarm       0.93      0.86      0.90       255
          clock_tick       0.92      0.81      0.86       245
      glass_breaking       0.20      0.80      0.32       255
          helicopter       0.79      0.60      0.68       228
            chainsaw       0.76      0.86      0.81       227
               siren       0.86      0.92      0.89       236
            car_horn       0.57      0.68      0.62       238
              engine       0.92      0.74      0.82       220
               train       0.64      0.62      0.63       242
        church_bells       0.84      0.96      0.89       245
            airplane       0.79      0.62      0.70       231
           fireworks       0.79      0.77      0.78       230
            hand_saw       0.89      0.89      0.89       243
    
           micro avg       0.71      0.71      0.71     12002
           macro avg       0.75      0.71      0.72     12002
        weighted avg       0.75      0.71      0.72     12002


​    


```python
import seaborn as sns
cn_matrix = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
plt.figure(figsize = (20,20))
sns.heatmap(cn_matrix, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd5a9b322e8>


