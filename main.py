"""
main.py: the base file to call the related class to train the model

"""

from util import get_dataset
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
    parser = argparse.ArgumentParser(description='Classify MNIST images.')
    parser.add_argument('-n',
        '--model_class_name',
        type=str,
        default=BEST_CLASS,
        help='Name of model class name')
    parser.add_argument('-i',
        '--model_id',
        type=str,
        default="000000",
        help='Model id which includes the model detail configurations')
    
    parser.add_argument('params', nargs='*')
    args = parser.parse_args()

    # Prepare cifar-100 data
    ds_train, ds_test = get_dataset()

    # set Logging 
    tf.logging.set_verbosity(tf.logging.INFO)

    # load model class by class file name
    ModelClass = import_class("model." + args.model_class_name)
    model = ModelClass(params=args.params)

    # train the model
    model.train_and_evaluate(ds_train, ds_test)

    