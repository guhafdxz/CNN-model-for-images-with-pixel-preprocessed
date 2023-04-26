# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 16:41:45 2022
****AlexNet CNN with Keras and cifar10 datasets***
Environment:
Environment:Intel CPU
python=3.9.X
tensorflow=2.10.0
numpy= 1.23.4
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import pandas as pd
import tensorflow.keras as keras
import tensorflow.keras.backend as K
tf.compat.v1.disable_eager_execution()
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,Input,Concatenate
from keras.models import Model


# Tensorflow-GPU config
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"



"Load caltech101 data"
train_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1,
    horizontal_flip=True,
    shear_range=0.2,
    width_shift_range=0.1
)
valid_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1
)
test_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1
)
#current work_direcory:./
df_train = pd.read_csv('./desc_caltech101_train.csv', encoding='utf8') #read csv file in the current work directory
df_valid=pd.read_csv('./desc_caltech101_valid.csv', encoding='utf8')
df_test=pd.read_csv('./desc_caltech101_test.csv', encoding='utf8')
df_train['class'] = df_train['class'].astype(str)
df_valid['class'] = df_valid['class'].astype(str)
df_test['class'] = df_test['class'].astype(str)
train_generator = train_gen.flow_from_dataframe(
    dataframe=df_train,
    directory="",
    x_col='file_name',
    y_col='class',
    target_size=(32, 32),
    batch_size=6907, 
    class_mode='categorical'
)
valid_generator = valid_gen.flow_from_dataframe(
    dataframe=df_valid,
    directory="",
    x_col='file_name',
    y_col='class',
    target_size=(32, 32),
    batch_size=820,
    class_mode='categorical'
)
test_generator = test_gen.flow_from_dataframe(
    dataframe=df_test,
    directory="",
    x_col='file_name',
    y_col='class',
    target_size=(32, 32),
    batch_size=950,
    class_mode='categorical'
)

def get_data(generator):    
    for step, (x, y) in enumerate(generator):
       print(x.shape)
       print(y.shape)  
       break
    return x,y
x_val,y_val=get_data(valid_generator)
x_test,y_test=get_data(test_generator)
x_train,y_train=get_data(train_generator)
# np.save('caltech_train_x',x_train)
# np.save('caltech_train_y',y_train)
# x_train=np.load('./caltech_train_x.npy')
# y_train=np.load('./caltech_train_y.npy')
xtest=x_test.astype('uint8')
xval=x_val.astype('uint8')
x_train_a=x_train.astype('uint8')
image_size=32  # replace with 224 if possible
'''
Image Pixel Processing
'''
def process_pixel(x_train,depth,size):
    if depth==32:
        return x_train
    xtrain=np.zeros((image_size,image_size),dtype='uint8')                     
    for i in range(0,size):
        g_xtrain=cv2.cvtColor(x_train[i], cv2.COLOR_RGB2GRAY) 
        if depth==8:
            xtrain=np.concatenate([xtrain,g_xtrain],axis=0) 
        if depth==4:
            xtrain=np.concatenate([xtrain,np.right_shift(g_xtrain,4)],axis=0)
        if depth==2:
            xtrain=np.concatenate([xtrain,np.right_shift(g_xtrain,6)],axis=0)
        if depth==1:
            _, b_xtrain= cv2.threshold(g_xtrain, 127, 255,cv2.THRESH_BINARY)  #binary
            xtrain=np.concatenate([xtrain,b_xtrain],axis=0)  
    xtrain=xtrain[image_size:].reshape(size,image_size,image_size)  
    return  xtrain



depth=2# pixel depth [1,2,4,8,32]
xtrain=process_pixel(x_train_a,depth,x_train_a.shape[0])
xval=process_pixel(xval,depth,xval.shape[0])
xtest=process_pixel(xtest,depth,xtest.shape[0])

xtrain= xtrain.astype('float32') / 255
xval = xval.astype('float32') / 255
xtest = xtest.astype('float32') / 255
if depth!=32: # channel=1
    xtrain=np.reshape(xtrain,(xtrain.shape[0],32,32,1))
    xval=np.reshape(xval,(xval.shape[0],32,32,1))
    xtest=np.reshape(xtest,(xtest.shape[0],32,32,1))


'''
VGG16
'''
# channel=3  #  3  for RGB image,1 for grey
# input_shape=(32,32,channel)
# def vgg16(channels):
#     model = Sequential()
#     model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
#     model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#     model.add(keras.layers.BatchNormalization())
#     model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#     model.add(keras.layers.BatchNormalization())
#     model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
#     model.add( Conv2D(256, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
#     model.add( MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(keras.layers.BatchNormalization())
#     model.add( Conv2D(512, (3, 3), padding='same', activation='relu'))
#     model.add( Conv2D(512, (3, 3), padding='same', activation='relu'))
#     model.add( Conv2D(512, (3, 3), padding='same', activation='relu'))
#     model.add( MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(keras.layers.BatchNormalization())
#     model.add( Conv2D(512, (3, 3), padding='same', activation='relu'))
#     model.add( Conv2D(512, (3, 3), padding='same', activation='relu'))
#     model.add(  Conv2D(512, (3, 3), padding='same', activation='relu'))
#     model.add( MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(keras.layers.BatchNormalization())
    
#     model.add( Flatten())
#     model.add( Dense(4096, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add( Dense(4096, activation='relu'))
#     model.add( Dense(1000, activation='softmax'))
#     model.add(Dense(101,activation='softmax'))      # output of dense_2
    
#     model.summary()
   
#     return model

imagesize=100
batch_size=50
epochs =20
channel=1  #  3  for RGB image,1 for grey
def alexnet(channel):
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(96, (11, 11), strides=(3, 3), input_shape=(imagesize,imagesize,channel), padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    # 2nd Convolutional Layer
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    # 3rd Convolutional Layer
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 4th Convolutional Layer
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 5th Convolutional Layer
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")) 
    
    #Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.3)) #0.3-0.5
    # 2nd Fully Connected Layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.3))
    # 3rd Fully Connected Layer
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(10, activation='softmax'))
    print(model.summary())

    return model
'''
Compile AlexNet and Train
'''
# model = vgg16(channel)
model=alexnet(channel)
# Compile network (loss function, optimizer, evaluation indicator)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Network training (training and validation data, training algebra,  training batch size)
# train_history = model.fit(xval, y_val, epochs=epochs, batch_size=batch_size, verbose=2)
train_history = model.fit(xtrain, y_train,validation_data=(xval, y_val), epochs=epochs, batch_size=batch_size, verbose=2)
# Sava model
model.save('alexnet_caltech101.h5')


"mixed different depth images to train"
# batch_size=300
# epochs =30
# x_val=x_val.astype('uint8')
# x_test=x_test.astype('uint8')
# xtrain_2=process_pixel(x_train_a[:1000],1,x_train_a[:1000].shape[0])
# xval_2=process_pixel(x_val[:500],1,x_val[:500].shape[0])
# xtest_2=process_pixel(x_test[:500],1,x_test[:500].shape[0])
# xtrain_2= xtrain_2.astype('float32') / 255
# xval_2= xval_2.astype('float32') / 255
# xtest_2= xtest.astype('float32') / 255
# xtrain_2=np.reshape(xtrain_2,(xtrain_2.shape[0],32,32,1))
# xval_2=np.reshape(xval_2,(xval_2.shape[0],32,32,1))
# xtest_2=np.reshape(xtest_2,(xtest_2.shape[0],32,32,1))
# concat_xval=np.concatenate([xval,xval_2],axis=0)
# concat_yval=np.concatenate([y_val,y_val[:500]],axis=0)
# train_history_2 = model.fit(np.concatenate([xtrain,xtrain_2],axis=0),np.concatenate([y_train,y_train[:1000]],axis=0),validation_data=(concat_xval, concat_yval), epochs=epochs, batch_size=batch_size, verbose=2)



'''
training process visualization function 
'''
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

# #The model output of on the test set
score = model.evaluate(xtest, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#  results of prediction
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)




   

