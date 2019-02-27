"""
main.py: the base file to call the related class to train the model

"""

from util import get_dataset, get_dataset2
import tensorflow as tf
import argparse
import os


BEST_CLASS="ModelCNN"

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify Cifar-100 images.')
    parser.add_argument('-n',
        '--model_class_name',
        type=str,
        default=BEST_CLASS,
        help='Name of model class name')
    parser.add_argument('-p',
        '--data_path',
        type=str,
        default="web",
        help='Model id which includes the model detail configurations')

    parser.add_argument('-g',
        '--img_augmentation',
        default=False,
        action="store_true",
        help='Activate image augmentation')

    parser.add_argument('-l',
        '--dynamic_learning_rate',
        default=False,
        action="store_true",
        help='Activate dynamic_learning_rate')

    parser.add_argument('-b',
        '--batch_size',
        default=128,
        type=int,
        help='batch size')

    parser.add_argument('params', nargs='*')
    args = parser.parse_args()

    # # Prepare cifar-100 data
    if args.data_path == 'web':
        ds_train, ds_test = get_dataset()
    else:
        ds_train, ds_test = get_dataset2()

    # set Logging 
    tf.logging.set_verbosity(tf.logging.INFO)

    # load model class by class file name
    ModelClass = import_class("model." + args.model_class_name)
    model = ModelClass(data_path=args.data_path, 
                       img_augmentation=args.img_augmentation,
                       dynamic_learning_rate=args.dynamic_learning_rate,
                       batch_size=args.batch_size,
                       params=args.params)

    # train the model
    model.train_and_evaluate(ds_train, ds_test)

    