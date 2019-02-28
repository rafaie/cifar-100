"""
autoencoder.py: An example of autoencoder trained by ImageNet dataset 
                and used for training of Cifar-100

"""

from util import get_dataset, get_dataset2
import tensorflow as tf
import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import cifar100

def load_imagenet_dataset():
    # cfair = "/work/cse496dl/shared/homework/02/"
    # train_data = np.load(cfair + 'imagenet_images.npy')
    # train_data = np.reshape(train_data, [-1, 32, 32, 3])

    (x_train, _), (x_test, _) = cifar100.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

    return x_train, x_test

def load_cifar100_dataset():



def get_autoencoder_model_vanilla():
    # Encoder
    inputSize = 3072
    outputSize = 3072
    hiddenSize = 128

    inputLayer = tf.keras.Input(shape = (inputSize,))
    encoderLayer = tf.keras.layers.Dense(hiddenSize, activation='relu')(inputLayer)
    decoderLayer = tf.keras.layers.Dense(outputSize, activation='sigmoid')(encoderLayer)

    autoencoder = tf.keras.Model(inputLayer, decoderLayer)



    # autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

def get_VanillaEncoderModel():
    inputSize = 3072
    outputSize = 3072
    hiddenSize = 128

    input_img = tf.keras.Input(shape=(inputSize,))
    encoded = tf.keras.layers.Dense(hiddenSize, activation='relu')(input_img)
    encoder = tf.keras.Model(input_img, encoded)

    return encoder

def get_VanillaDecoderModel(autoEncoder):
    inputSize = 3072
    outputSize = 3072
    hiddenSize = 128

    encoderInput= tf.keras.Input(shape=(hiddenSize,))
    decoderLayer = autoEncoder.layers[-1]
    decoder = tf.keras.Model(encoderInput, decoderLayer(encoderInput))

    return decoder

def get_autoencoder_model_multilayer():
    pass

def get_autoencoder_model_convolutional():

    ## 28 28 1
    input = tf.keras.Input(shape=(32, 32, 3))

    # Encoder
    firstEncodeConvLayer = tf.keras.layers.Conv2D(8, (5, 5), activation='relu', padding='same')(input)
    firstPoolLayer = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(firstEncodeConvLayer)
    secondEncodeConvLayer = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(firstPoolLayer)
    secondPoolLayer = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(secondEncodeConvLayer)
    thirdEncodeConvLayer = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(secondPoolLayer)
    encoded = tf.keras.layers.MaxPooling2D((4, 4), padding='same')(thirdEncodeConvLayer)

    # Decoder
    firstDecodeConvLayer = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(encoded)
    firstUp= tf.keras.layers.UpSampling2D((2, 2))(firstDecodeConvLayer)
    secondDecodeConvLayer = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(firstUp)
    secondUp = tf.keras.layers.UpSampling2D((2, 2))(secondDecodeConvLayer)
    thirdDecodeConvLayer= tf.keras.layers.Conv2D(3, (5, 5), activation='relu')(secondUp)
    thirdUp = tf.keras.layers.UpSampling2D((2, 2))(thirdDecodeConvLayer)
    decoded = tf.keras.layers.Conv2D(3, (5, 5), activation='sigmoid', padding='same')(thirdUp)

    convAutoEncoder = tf.keras.Model(input, thirdDecodeConvLayer)
    return convAutoEncoder


def get_autoencoder_model_regularized():
    pass

def plotImages(testData, decodedImages):
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(testData[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decodedImages[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def train_autoencoders():

    # vanillaAutoEncoder = get_autoencoder_model_vanilla()
    # vanillaAutoEncoder.compile(optimizer='adam', loss='mse')
    #
    trainData, testData = load_imagenet_dataset()
    #
    # vanillaAutoEncoder.fit(trainData, trainData, epochs=50, batch_size=256, shuffle=True, validation_data = (testData, testData))
    #
    # encoder = get_VanillaEncoderModel()
    # decoder = get_VanillaDecoderModel(vanillaAutoEncoder)



    convAutoEncoder = get_autoencoder_model_convolutional()
    convAutoEncoder.compile(optimizer='adam', loss='mse')
    convAutoEncoder.fit(trainData, trainData, epochs=5, batch_size=256, shuffle=True, validation_data = (testData, testData))

    decodedImages = convAutoEncoder.predict(testData)
    plotImages(testData, decodedImages)



    pass


def reuse_encoders():
    pass


train_autoencoders()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='autoencoders')
#     parser.add_argument('-action',
#         type=str,
#         help='action [train|reuse]')
#
#     args = parser.parse_args()
#     if args.action == 'train':
#         train_autoencoders()
#     elif args.action == 'reuse':
#         reuse_encoders()
#     else:
#         print('Please use with this param: -action [train|reuse]')
#