"""
autoencoder.py: An example of autoencoder trained by ImageNet dataset 
                and used for training of Cifar-100

"""

from util import get_dataset, get_dataset2
import tensorflow as tf
import argparse
import os


def load_imagenet_dataset():
    pass

def load_cifar100_dataset():
    pass

def get_autoencoder_model_vanilla():
    pass

def get_autoencoder_model_multilayer():
    pass

def get_autoencoder_model_convolutional():
    pass

def get_autoencoder_model_regularized():
    pass


def train_autoencoders():
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
 