"""
autoencoder.py: An example of autoencoder trained by ImageNet dataset 
                and used for training of Cifar-100

"""

from util import get_dataset, get_dataset2
from model import BaseConfig, BaseModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from operator import mul
from functools import reduce
import argparse
import os
import util
import sys


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



def conv_block(inputs, filters, downscale=1):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: int
    """
    with tf.name_scope('conv_block') as scope:
        conv = tf.layers.conv2d(inputs, filters, 3, 1, padding='same')
        down_conv = tf.layers.conv2d(conv, filters, 3, strides=downscale, padding='same')
        return down_conv

def gaussian_encoder(inputs, latent_size):
    """inputs should be a tensor of images whose height and width are multiples of 4"""
    x = conv_block(inputs, 8, downscale=2)
    x = conv_block(x, 16, downscale=2)
    mean = tf.layers.dense(x, latent_size)
    log_scale = tf.layers.dense(x, latent_size)
    return mean, log_scale


def gaussian_sample(mean, log_scale):
    # noise is zero centered and std. dev. 1
    gaussian_noise = tf.random_normal(shape=tf.shape(mean))
    return mean + (tf.exp(log_scale) * gaussian_noise)


def upscale_block2(x, scale=2, name='upscale_block'):
    """[Sub-Pixel Convolution](https://arxiv.org/abs/1609.05158) """
    n, w, h, c = x.get_shape().as_list()
    x = tf.layers.conv2d(x, c * scale ** 2, (3, 3), activation=tf.nn.relu, padding='same', name=name)
    output = tf.depth_to_space(x, scale)
    return output

def decoder(inputs, output_shape):
    """output_shape should be a length 3 iterable of ints"""
    h, w, c = output_shape
    initial_shape = [h // 4, w // 4, c]
    initial_size = reduce(mul, initial_shape)
    x = tf.layers.dense(inputs, initial_size // 64, name='decoder_dense')
    x = tf.reshape(x, [-1] + initial_shape)
    x = upscale_block2(x, name='upscale1')
    return upscale_block2(x, name='upscale2')

# define an epsilon
EPS = 1e-10
def std_gaussian_KL_divergence(mu, log_sigma):
    """Analytic KL distance between N(mu, e^log_sigma) and N(0, 1)"""
    sigma = tf.exp(log_sigma)
    return -0.5 * tf.reduce_sum(
        1 + tf.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma), 1)


def flatten(inputs):
    """
    Flattens a tensor along all non-batch dimensions.
    This is correctly a NOP if the input is already flat.
    """
    if len(shape(inputs)) == 2:
        return inputs
    else:
        size = inputs.get_shape().as_list()[1:]
        return tf.reshape(inputs, [-1, size])

def bernoulli_logp(alpha, sample):
    """Calculates log prob of sample under bernoulli distribution.
    
    Note: args must be in range [0,1]
    """
    alpha = flatten(alpha)
    sample = flatten(sample)
    return tf.reduce_sum(sample * tf.log(EPS + alpha) +
                         ((1 - sample) * tf.log(EPS + 1 - alpha)), 1)

def discretized_logistic_logp(mean, logscale, sample, binsize=1 / 256.0):
    """Calculates log prob of sample under discretized logistic distribution."""
    scale = tf.exp(logscale)
    sample = (tf.floor(sample / binsize) * binsize - mean) / scale
    logp = tf.log(
        tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + EPS)

    if logp.shape.ndims == 4:
        logp = tf.reduce_sum(logp, [1, 2, 3])
    elif logp.shape.ndims == 2:
        logp = tf.reduce_sum(logp, 1)
    return logp


def vae_loss(inputs, outputs, latent_mean, latent_log_scale, output_dist, output_log_scale=None):
    """Calculate the VAE loss (aka [ELBO](https://arxiv.org/abs/1312.6114))
    
    Args:
        - inputs: VAE input
        - outputs: VAE output
        - latent_mean: parameter of latent distribution
        - latent_log_scale: log of std. dev. of the latent distribution
        - output_dist: distribution parameterized by VAE output, must be in ['logistic', 'bernoulli']
        - output_log_scale: log scale parameter of the output dist if it's logistic, can be learnable
        
    Note: output_log_scale must be specified if output_dist is logistic
    """
    # Calculate reconstruction loss
    # Equal to minus the log likelihood of the input data under the VAE's output distribution
    if output_dist == 'bernoulli':
        outputs = tf.sigmoid(outputs)
        reconstruction_loss = -bernoulli_logp(outputs, inputs)
    elif output_dist == 'logistic':
        outputs = tf.clip_by_value(outputs, 1 / 512., 1 - 1 / 512.)
        reconstruction_loss = -discretized_logistic_logp(outputs, output_log_scale, inputs)
    else:
        print('Must specify an argument for output_dist in [bernoulli, logistic]')
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        
    # Calculate latent loss
    latent_loss = std_gaussian_KL_divergence(latent_mean, latent_log_scale)
    latent_loss = tf.reduce_mean(latent_loss)
    
    return reconstruction_loss, latent_loss

def get_autoencoder_vae_logistic(img_shape=[32, 32, 3], latent_size=3):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None] + img_shape, name='input_placeholder')
    # VAE
    means, log_scales = gaussian_encoder(inputs, latent_size)
    codes = gaussian_sample(means, log_scales)
    codes = tf.identity(codes, name='codes')
    outputs = decoder(codes, img_shape)
    # with tf.variable_scope("model", reuse=True) as scope:
    #     gen_sample = decoder(codes, img_shape)

    # calculate loss with learnable parameter for output log_scale
    output_log_scale = tf.get_variable("output_log_scale", initializer=tf.constant(0.0, shape=img_shape))
    print(inputs.shape, outputs.shape)
    reconstruction_loss, latent_loss = vae_loss(inputs, outputs, means, log_scales, 'logistic', output_log_scale)

    total_loss = reconstruction_loss + latent_loss

    return inputs, codes, outputs, total_loss, output_log_scale


def get_autoencoder_sparse(shape=[32, 32, 3], sparsity_weight=5e-3, code_size=192):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None] + shape, name='input_placeholder')
    encoder_16 = downscale_block(x)
    encoder_8 = downscale_block(encoder_16)
    flatten_dim = np.prod(encoder_8.get_shape().as_list()[1:])
    flat = tf.reshape(encoder_8, [-1, flatten_dim])
    code = tf.layers.dense(flat, code_size, activation=tf.nn.relu, name='codes_id')
    code = tf.identity(code, name='codes')
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


def get_autoencoder_auto_noise(shape=[32, 32, 3], sparsity_weight=5e-3, code_size=192, noise_level = 0.1):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None] + shape, name='input_placeholder')
    x_noisy = x + noise_level * tf.random_normal(tf.shape(x))
    encoder_16 = downscale_block(x_noisy)
    encoder_8 = downscale_block(encoder_16)
    flatten_dim = np.prod(encoder_8.get_shape().as_list()[1:])
    flat = tf.reshape(encoder_8, [-1, flatten_dim])
    code = tf.layers.dense(flat, code_size, activation=tf.nn.relu, name='codes_id')
    code = tf.identity(code, name='codes')
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

def save_img_reconstruction(k, img1, img2, outputs_out, dir='imgs'):
    os.makedirs(dir, exist_ok=True)

    plt.imshow(np.squeeze(img1) + 128)
    plt.savefig(os.path.join(dir, k + '_sample1'))
    plt.clf()

    plt.imshow(np.squeeze(img2) )
    plt.savefig(os.path.join(dir, k + '_sample2'))
    plt.clf()
    
    fig=plt.figure(figsize=(10, 10))
    columns = 3
    rows = 3
    for i in range(rows):
        for j in range(columns):
            if i == j:
                img = np.squeeze(outputs_out[i] + 128)
            else:
                img = np.squeeze(outputs_out[i]) - np.squeeze(outputs_out[j] + 128)
            fig.add_subplot(columns, rows, (i*rows) + j + 1)
            plt.imshow(img)
    plt.savefig(os.path.join(dir, k + '_econstruction_sample'))
    plt.clf()

encoder_funcs = {'autoencoder_sparse': get_autoencoder_sparse,
                 'autoencoder_auto_noise': get_autoencoder_auto_noise,
                 'autoencoder_vae_logistic': get_autoencoder_vae_logistic}

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
        for epoch in range(100):
            print("Start epoch :", epoch)
            for i in range(train_data.shape[0] // batch_size):
                batch_xs = train_data[i*batch_size:(i+1)*batch_size, :]
                session.run(train_op, {inputs: batch_xs})
        print("Training of the " + k + " autoencoder is done!")

        # save 
        idx = np.random.randint(train_data.shape[0])
        inputs_data = np.repeat(np.expand_dims(train_data[idx], axis=0), 3, axis=0)
        inputs_out, output_log_scale, outputs_out = session.run([inputs, output_log_scale, outputs], {inputs: inputs_data})

        save_img_reconstruction(k, train_data[0], inputs_out[0], outputs_out)

        # save model
        save_model(session, k)

def load_autoencoder(autoenconder_name ):
    session = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join('auto_models', autoenconder_name + '.meta'))
    saver.restore(session, os.path.join('auto_models', autoenconder_name))
    graph = session.graph
    input_placeholder = graph.get_tensor_by_name('input_placeholder:0')
    code = graph.get_tensor_by_name('codes:0')
    print(code.shape)
    if autoenconder_name is not 'autoencoder_vae_logistic':
        code = tf.reshape(code, [-1, 8, 8, 3])
    return input_placeholder, code


def get_model(autoenconder_name):
    
    input_placeholder, code = load_autoencoder(autoenconder_name)

    features = util.conv_layers(code,
                                   filters=[64, 192, 384, 256, 256],
                                   kernels=[3, 3, 3, 3, 3],
                                   pool_sizes=[2, 2, 2, 2, 2])
    features = tf.contrib.layers.flatten(features)

    logits = util.dense_layers(
                features, [512, 100],
                drop_rates=0.5,
                linear_top_layer=True)

    logits = tf.identity(logits, name='output')

    labels = tf.placeholder(tf.int64, [None], name='label')

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = loss #+ 0.02 * sum(reg_losses)

    return logits, total_loss, input_placeholder, labels

def evaluate(test_images, test_labels, total_loss, output, session,
                 x, y, batch_size, test_num_examples):
    ce_vals = []
    conf_mxs = []

    c_pred = tf.cast(tf.equal(y, 
                                tf.argmax(output, axis=1)), tf.int32)
    c_preds = []
    for i in range(test_num_examples // batch_size):
        batch_xs = test_images[i * batch_size:(i + 1) * batch_size]
        batch_ys = test_labels[i * batch_size:(i + 1) * batch_size]
        test_ce,  c_pred_val= session.run(
            [tf.reduce_mean(total_loss), c_pred], {
                x: batch_xs,
                y: batch_ys
            })
        ce_vals.append(test_ce)
        c_preds += c_pred_val.tolist()
    return (ce_vals, conf_mxs, sum(c_preds)/len(c_preds))

def train_and_evaluate(autoenconder_name, train_images, train_labels, test_images, test_labels):
    train_num_examples = train_images.shape[0]
    test_num_examples = test_images.shape[0]

    output, total_loss, x, y = get_model(autoenconder_name)
    
 
    with tf.Session() as session:
        with tf.variable_scope("model_1") as scope:
            global_step_tensor = tf.get_variable(
                    'global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
            session.run(tf.global_variables_initializer())
            batch_size = 120
            ce_vals = []
            print(train_num_examples // batch_size)
            for j in range(500):
                for i in range(train_num_examples // batch_size):
                    batch_xs = train_images[i * batch_size:(i + 1) * batch_size]
                    batch_ys = train_labels[i * batch_size:(i + 1) * batch_size]
                    _, train_ce = session.run(
                        [train_op, tf.reduce_mean(total_loss)], {
                            x: batch_xs,
                            y: batch_ys
                        })
                    ce_vals.append(train_ce)
                    avg_train_ce = sum(ce_vals) / len(ce_vals)

                    if i % 50 == 0 and i > 10:
                        print('epoch:', j, ',step:', i, ', TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
                        ce_vals, conf_mxs, acc = evaluate(test_images, test_labels, total_loss, output, session,
                                                          x, y, batch_size, test_num_examples)
                        avg_test_ce = sum(ce_vals) / len(ce_vals)
                        print('------------------------------')
                        print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
                        print('TEST Accuracy     : ' + str(acc))
                        print(','.join(['TEST_PROGRESS', autoenconder_name, str(j), str(i), str(avg_train_ce), 
                                            str(avg_test_ce), str(acc)]))





def reuse_encoders():
    counter = 0
    train_x, train_y, test_x, test_y = util.prepare_data()

    for k, f in encoder_funcs.items():
        print('Start reusing the ' + k + ' autoencoder!')
        # train the model
        train_and_evaluate(k, train_x, train_y, test_x, test_y)
        sys.exit(0)



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
    elif args.action == 'all':
        train_autoencoders()
        reuse_encoders()
    else:
        print('Please use with this param: -action [train|reuse|all]')
 