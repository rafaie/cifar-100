"""
autoencoder.py: An example of autoencoder trained by ImageNet dataset 
                and used for training of Cifar-100

"""

from util import get_dataset, get_dataset2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import argparse
import os
import util


def load_imagenet_dataset(data_dir='data'):
    images = np.load(os.path.join(data_dir, 'imagenet_images.npy'))
    print('Imagenet dataset size', images.shape)
    return images


def get_size(shape):
    s = 1
    for i in len(shape):
        s *= shape[i]
    return s


def load_cifar100_dataset():
    pass


def upscale_block(x, scale=2):
    """transpose convolution upscale"""
    return tf.layers.conv2d_transpose(x, 1, 3, strides=(scale, scale), padding='same', activation=tf.nn.relu)

def downscale_block(x, scale=2):
    _, _, _, c = x.get_shape().as_list()
    return tf.layers.conv2d(x, np.floor(c * 1.25), 3, strides=scale, padding='same')


def get_autoencoder_sparse(shape=[32, 32, 3], sparsity_weight=5e-3, code_size=100):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None] + shape, name='input')
    encoder_16 = downscale_block(x)
    encoder_8 = downscale_block(encoder_16)
    flatten_dim = np.prod(encoder_8.get_shape().as_list()[1:])
    flat = tf.reshape(encoder_8, [-1, flatten_dim])
    code = tf.layers.dense(flat, code_size, activation=tf.nn.relu, name='code')
    hidden_decoder = tf.layers.dense(code, 64, activation=tf.nn.elu)
    decoder_8 = tf.reshape(hidden_decoder, [-1, 8, 8, 1])
    decoder_16 = upscale_block(decoder_8)
    output = upscale_block(decoder_16)

    # calculate loss
    sparsity_loss = tf.norm(code, ord=1, axis=1)
    reconstruction_loss = tf.reduce_mean(tf.square(output - x)) # Mean Square Error
    total_loss = reconstruction_loss + sparsity_weight * sparsity_loss

    # output_log_scale
    output_log_scale = tf.get_variable("output_log_scale", initializer=tf.constant(0.0, shape=shape))

    return x, code, output, total_loss, output_log_scale

def save_model(session, k, model_dir='auto_models'):
    os.makedirs(model_dir, exist_ok=True)
    tf.train.Saver().save(session, os.path.join(model_dir, k))

def save_img_reconstruction(k, img, outputs_out, dir='imgs'):
    os.makedirs(dir, exist_ok=True)
    plt.imshow(np.squeeze(img))
    plt.savefig(os.path.join(dir, k + '_sample'))

    fig=plt.figure(figsize=(10, 10))
    columns = 3
    rows = 3
    for i in range(rows):
        for j in range(columns):
            if i == j:
                img = np.squeeze(outputs_out[i])
            else:
                img = np.squeeze(outputs_out[i]) - np.squeeze(outputs_out[j])
            fig.add_subplot(columns, rows, (i*rows) + j + 1)
            plt.imshow(img)
    plt.savefig(os.path.join(dir, k + '_econstruction_sample'))

encoder_funcs = {'autoencoder_sparse': get_autoencoder_sparse}
def train_autoencoders():
    train_data = load_imagenet_dataset()

    for k, f in encoder_funcs.items():
        print('Start training the ' + k + ' autoencoder!')
        inputs, code, outputs, total_loss, output_log_scale = f()

        # setup optimizer
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(total_loss)

        # train for an epoch
        batch_size = 16
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        for epoch in range(1):
            print("Start epoch :", epoch)
            for i in range(train_data.shape[0] // batch_size)[:100]:
                batch_xs = train_data[i*batch_size:(i+1)*batch_size, :]
                session.run(train_op, {inputs: batch_xs})
        print("Training of the " + k + " autoencoder is done!")

        # save 
        idx = np.random.randint(train_data.shape[0])
        inputs_data = np.repeat(np.expand_dims(train_data[idx], axis=0), 3, axis=0)
        inputs_out, output_log_scale, outputs_out = session.run([inputs, output_log_scale, outputs], {inputs: inputs_data})

        save_img_reconstruction(k, inputs_out[0], outputs_out)

        # save model
        save_model(session, k)


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
 