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
from keras import regularizers

def load_imagenet_dataset():
    # cfair = "/work/cse496dl/shared/homework/02/"
    # train_data = np.load(cfair + 'imagenet_images.npy')
    # train_data = np.reshape(train_data, [-1, 32, 32, 3])

    (x_train, _), (x_test, _) = cifar100.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # adapt this if using `channels_first` image data format
    # x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

    return x_train, x_test

def normalizedData():
    (x_train, _), (x_test, _) = cifar100.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

    return x_train, x_test

def load_cifar100_dataset():
    pass


def get_autoencoder_model_vanilla():
    # Encoder
    inputSize = 3072
    outputSize = 3072
    hiddenSize = 128

    inputLayer = tf.keras.Input(shape = (inputSize,))
    encoderLayer = tf.keras.layers.Dense(hiddenSize, activation='relu')(inputLayer)
    decoderLayer = tf.keras.layers.Dense(outputSize, activation='sigmoid')(encoderLayer)
    autoencoder = tf.keras.Model(inputLayer, decoderLayer)

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
    inputSize = 3072
    outputSize = 3072
    hiddenSize = 1539
    codeSize = 128

    input = tf.keras.layers.Input(shape=(inputSize,))


    firstDense = tf.keras.layers.Dense(hiddenSize, activation='relu')(input)
    encoder = tf.keras.layers.Dense(codeSize, activation='relu')(firstDense)

    secondDense = tf.keras.layers.Dense(hiddenSize, activation='relu')(encoder)
    decoder = tf.keras.layers.Dense(outputSize, activation='sigmoid')(secondDense)

    multiLayerAutoEncoder = tf.keras.Model(input, decoder)

    return multiLayerAutoEncoder

def get_autoencoder_model_convolutional():


    input = tf.keras.Input(shape=(32, 32, 3))

    # Encoder
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), activation='relu')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    convAutoEncoder = tf.keras.Model(input, decoded)

    return convAutoEncoder


def get_autoencoder_model_regularized_Sparse():
    inputSize = 3072
    outputSize = 3072
    hiddenSize = 128

    input = tf.keras.layers.Input(shape=(inputSize,))

    encoder = tf.keras.layers.Dense(hiddenSize, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input)
    decoder = tf.keras.layers.Dense(outputSize, activation='sigmoid')(encoder)

    autoencoder = tf.keras.Model(encoder, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')


    pass

def plotImages(testData, decodedImages, name):
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

    plt.savefig(('/autoEncoders/' + name + '/img/comparisonPicture'))



def train_save_singleAutoEncoder(name, e, b):

    trainData, testData = load_imagenet_dataset()
    if(name == 'conv'):
        autoEncoder = get_autoencoder_model_convolutional()
        trainData, testData = normalizedData()
    elif name == 'sparse':
        autoEncoder = get_autoencoder_model_regularized_Sparse()
    elif name == 'multi':
        autoEncoder = get_autoencoder_model_multilayer()
    elif name == 'vanilla':
        autoEncoder = get_autoencoder_model_vanilla()


    autoEncoder.compile(optimizer='adam', loss='mse')
    autoEncoder.fit(trainData, trainData, epochs=e, batch_size=b, shuffle=True,
                        validation_data=(testData, testData))

    tf.keras.models.save_model(
        autoEncoder,
        ('/autoEncoders/' + name),
        overwrite=True,
        include_optimizer=True
    )
    decodedImages = autoEncoder.predict(testData)
    plotImages(testData, decodedImages, name)





def train_autoencoders():

    train_save_singleAutoEncoder('conv', 1, 256)
    train_save_singleAutoEncoder('sparse', 1, 256)
    train_save_singleAutoEncoder('vanilla', 1, 256)
    train_save_singleAutoEncoder('multi', 1, 256)

    pass

def reuse_encoders():
    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='autoencoders')
    parser.add_argument('-action',
        type=str,
        help='action [train|reuse]')

    args = parser.parse_args()
    if args.action == 'train':
        train_autoencoders()
    elif args.action == 'reuse':
        reuse_encoders()
    else:
        print('Please use with this param: -action [train|reuse]')
