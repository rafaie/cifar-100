"""
util.py: it includes Helper functions
          Thanks Vahid Kazemi for sharing the cifar-100 training data extration
          https://github.com/vahidk/TensorflowFramework/blob/master/dataset/cifar100.py

"""

import tensorflow as tf
import numpy as np
import os
import tarfile
import shutil
import pickle
import functools
import numbers

from six.moves import urllib


tf.logging.set_verbosity(tf.logging.INFO)  # if you want to see the log info


REMOTE_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
ARCHIVE_NAME = "cifar-100-python.tar.gz"
FILE_LIST=['file.txt~', 'meta', 'test', 'train']
DATA_DIR = "data"
TRAIN_BATCHES = ["train"]
TEST_BATCHES = ["test"]

IMAGE_SIZE = 32
NUM_CLASSES = 100


# Utils to load dataset
def download_data():
     """Download the cifar 100 dataset."""
     if not os.path.exists(DATA_DIR):
          os.makedirs(DATA_DIR)
     if not os.path.exists(os.path.join(DATA_DIR ,ARCHIVE_NAME)):
          print("Downloading...")
          urllib.request.urlretrieve(REMOTE_URL, os.path.join(DATA_DIR ,ARCHIVE_NAME))

     for f in FILE_LIST:
          if not os.path.exists(os.path.join(DATA_DIR ,f)):
               for f2 in FILE_LIST:
                    if os.path.exists(os.path.join(DATA_DIR ,f2)):
                         os.remove(os.path.join(DATA_DIR ,f2))

               print("Extracting files...")
               tar = tarfile.open(os.path.join(DATA_DIR ,ARCHIVE_NAME))
               tar.extractall(DATA_DIR)
               tar.close()

               for f in os.listdir(os.path.join(DATA_DIR ,ARCHIVE_NAME.split('.')[0])):
                    shutil.move(os.path.join(DATA_DIR ,ARCHIVE_NAME.split('.')[0], f), DATA_DIR)
               
               os.rmdir(os.path.join(DATA_DIR ,ARCHIVE_NAME.split('.')[0]))
               break

def load_data():
     # Load train data 
     with open(os.path.join(DATA_DIR , 'train'), 'rb') as fo:
          train_data = pickle.load(fo, encoding='bytes')
          train_x = train_data[b'data'].reshape((len(train_data[b'data']), 3, 32, 32))
          train_x = train_x.transpose(0, 2, 3, 1).astype(np.float32)
          train_y = np.array(train_data[b'fine_labels'], dtype=np.int32)
          print(train_x.dtype)

     # Load test data 
     with open(os.path.join(DATA_DIR , 'test'), 'rb') as fo:
          test_data = pickle.load(fo, encoding='bytes')
          test_x = test_data[b'data'].reshape((len(test_data[b'data']), 3, 32, 32))
          test_x = test_x.transpose(0, 2, 3, 1).astype(np.float32)
          test_y = np.array(test_data[b'fine_labels'], dtype=np.int32)

     print(train_x.shape, test_x.shape)
     return (train_x, train_y, test_x, test_y)

def prepare_data():
     download_data()
     return load_data()


def get_dataset():
     train_x, train_y, test_x, test_y = prepare_data()

     ds_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
     ds_test  = tf.data.Dataset.from_tensor_slices((test_x, test_y))

     return (ds_train, ds_test)


# Operation Util class copied from this project: https://github.com/vahidk/TensorflowFramework

#https://github.com/vahidk/TensorflowFramework/blob/master/common/ops/activation_ops.py
def leaky_relu(tensor, alpha=0.1):
  """Computes the leaky rectified linear activation."""
  return tf.maximum(tensor, alpha * tensor)



# https://github.com/vahidk/TensorflowFramework/blob/master/common/ops/norm_ops.py
def batch_normalization(tensor, training=False, epsilon=0.001, momentum=0.9,
                        fused_batch_norm=False, name=None):
  """Performs batch normalization on given 4-D tensor.
  The features are assumed to be in NHWC format. Noe that you need to
  run UPDATE_OPS in order for this function to perform correctly, e.g.:
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = optimizer.minimize(loss)
  Based on: https://arxiv.org/abs/1502.03167
  """
  with tf.variable_scope(name, default_name="batch_normalization"):
    channels = tensor.shape.as_list()[-1]
    axes = list(range(tensor.shape.ndims - 1))

    beta = tf.get_variable(
      "beta", channels, initializer=tf.zeros_initializer())
    gamma = tf.get_variable(
      "gamma", channels, initializer=tf.ones_initializer())

    avg_mean = tf.get_variable(
      "avg_mean", channels, initializer=tf.zeros_initializer(),
      trainable=False)
    avg_variance = tf.get_variable(
      "avg_variance", channels, initializer=tf.ones_initializer(),
      trainable=False)

    if training:
      if fused_batch_norm:
        mean, variance = None, None
      else:
        mean, variance = tf.nn.moments(tensor, axes=axes)
    else:
      mean, variance = avg_mean, avg_variance

    if fused_batch_norm:
      tensor, mean, variance = tf.nn.fused_batch_norm(
        tensor, scale=gamma, offset=beta, mean=mean, variance=variance,
        epsilon=epsilon, is_training=training)
    else:
      tensor = tf.nn.batch_normalization(
        tensor, mean, variance, beta, gamma, epsilon)

    if training:
      update_mean = tf.assign(
        avg_mean, avg_mean * momentum + mean * (1.0 - momentum))
      update_variance = tf.assign(
        avg_variance, avg_variance * momentum + variance * (1.0 - momentum))

      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance)

  return tensor

# https://github.com/vahidk/TensorflowFramework/blob/master/common/ops/merge_ops.py
def merge(tensors, units, activation=tf.nn.relu, name=None, **kwargs):
  """Merge tensors with broadcasting support."""
  with tf.variable_scope(name, default_name="merge"):
    projs = []
    for i, tensor in enumerate(tensors):
      proj = tf.layers.dense(
          tensor, units, name="proj_%d" % i, **kwargs)
      projs.append(proj)

    result = projs.pop()
    for proj in projs:
      result = result + proj

    if activation:
      result = activation(result)
  return result


# https://github.com/vahidk/TensorflowFramework/blob/master/common/ops/layers_ops.py
def dense_layers(tensor,
                 units,
                 activation=tf.nn.relu,
                 use_bias=True,
                 linear_top_layer=False,
                 drop_rates=None,
                 batch_norm=False,
                 training=False,
                 weight_decay=0.0002,
                 **kwargs):
  """Builds a stack of fully connected layers with optional dropout."""
  if drop_rates is None:
    drop_rates = [0.] * len(units)
  elif isinstance(drop_rates, numbers.Number):
    drop_rates = [drop_rates] * len(units)
  for i, (size, drp) in enumerate(zip(units, drop_rates)):
    if i == len(units) - 1 and linear_top_layer:
      activation = None
    with tf.variable_scope("dense_block_%d" % i):
      tensor = tf.layers.dropout(tensor, drp)
      tensor = tf.layers.dense(
        tensor, size, use_bias=use_bias,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.glorot_uniform_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        **kwargs)
      if activation:
        if batch_norm:
          tensor = batch_normalization(tensor, training=training)
        tensor = activation(tensor)
  return tensor

# https://github.com/vahidk/TensorflowFramework/blob/master/common/ops/metric_ops.py
def interocular_ratio(labels, predictions, left_eye, right_eye, name=None):
  with tf.name_scope(name, default_name="interocular_ratio"):
    labels_t = tf.transpose(labels, [1, 0, 2])
    left_eye_pos = tf.reduce_mean(tf.gather(labels_t, left_eye), axis=0)
    right_eye_pos = tf.reduce_mean(tf.gather(labels_t, right_eye), axis=0)
    landmarks_l2 = tf.norm(predictions - labels, axis=2)
    eyes_l2 = tf.norm(left_eye_pos - right_eye_pos, axis=1)
    ratio = landmarks_l2 / tf.expand_dims(eyes_l2, axis=1)
    return ratio


def top_k_error(labels, predictions, k, name=None):
  with tf.name_scope(name, default_name="top_k_error"):
    labels = tf.expand_dims(tf.to_int32(labels), axis=-1)
    _, top_k = tf.nn.top_k(predictions, k=k)
    in_top_k = tf.reduce_mean(tf.to_float(tf.equal(top_k, labels)), -1)
    return 1 - in_top_k


def conv_layers(tensor,
                filters,
                kernels,
                strides=None,
                pool_sizes=None,
                pool_strides=None,
                padding="same",
                activation=tf.nn.relu,
                use_bias=False,
                linear_top_layer=False,
                drop_rates=None,
                conv_method="conv",
                pool_method="conv",
                pool_activation=None,
                batch_norm=False,
                training=False,
                weight_decay=0.0002,
                **kwargs):
  """Builds a stack of convolutional layers with dropout and max pooling."""
  if pool_sizes is None:
    pool_sizes = [1] * len(filters)
  if pool_strides is None:
    pool_strides = pool_sizes
  if strides is None:
    strides = [1] * len(filters)
  if drop_rates is None:
    drop_rates = [0.] * len(filters)
  elif isinstance(drop_rates, numbers.Number):
    drop_rates = [drop_rates] * len(filters)

  if conv_method == "conv":
    conv = functools.partial(
      tf.layers.conv2d,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
  elif conv_method == "transposed":
    conv = functools.partial(
      tf.layers.conv2d_transpose,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
  elif conv_method == "separable":
    conv = functools.partial(
      tf.layers.separable_conv2d,
      depthwise_initializer=tf.glorot_uniform_initializer(),
      pointwise_initializer=tf.glorot_uniform_initializer(),
      depthwise_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      pointwise_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

  for i, (fs, ks, ss, pz, pr, drp) in enumerate(
    zip(filters, kernels, strides, pool_sizes, pool_strides, drop_rates)):
    with tf.variable_scope("conv_block_%d" % i):
      if i == len(filters) - 1 and linear_top_layer:
        activation = None
        pool_activation = None
      tensor = tf.layers.dropout(tensor, drp)
      tensor = conv(
        tensor, fs, ks, ss, padding, use_bias=use_bias, name="conv2d",
        **kwargs)
      if activation:
        if batch_norm:
          tensor = batch_normalization(tensor, training=training)
        tensor = activation(tensor)
      if pz > 1:
        if pool_method == "max":
          tensor = tf.layers.max_pooling2d(
            tensor, pz, pr, padding, name="max_pool")
        elif pool_method == "std":
          tensor = tf.space_to_depth(tensor, pz, name="space_to_depth")
        elif pool_method == "dts":
          tensor = tf.depth_to_space(tensor, pz, name="depth_to_space")
        else:
          tensor = conv(
            tensor, fs, pz, pr, padding, use_bias=use_bias,
            name="strided_conv2d", **kwargs)
          if pool_activation:
            if batch_norm:
              tensor = batch_normalization(tensor, training=training)
            tensor = pool_activation(tensor)
  return tensor


def merge_layers(tensors, units, activation=tf.nn.relu,
                 linear_top_layer=False, drop_rates=None,
                 name=None, **kwargs):
  """Merge followed by a stack of dense layers."""
  if drop_rates is None:
    drop_rates = [0.] * len(units)
  elif isinstance(drop_rates, numbers.Number):
    drop_rates = [drop_rates] * len(units)
  with tf.variable_scope(name, default_name="merge_layers"):
#     result = tf.layers.dropout(tensors, drop_rates[0])
    result = merge(tensors, units[0], activation, **kwargs)
    result = dense_layers(result, units[1:],
                          activation=activation,
                          drop_rates=drop_rates[1:],
                          linear_top_layer=linear_top_layer,
                          **kwargs)
  return result


def squeeze_and_excite(tensor, ratio, name=None):
  """Apply squeeze/excite on given 4-D tensor.
  Based on: https://arxiv.org/abs/1709.01507
  """
  with tf.variable_scope(name, default_name="squeeze_and_excite"):
    original = tensor
    units = tensor.shape.as_list()[-1]
    tensor = tf.reduce_mean(tensor, [1, 2], keepdims=True)
    tensor = dense_layers(
      tensor, [units / ratio, units], use_bias=False, linear_top_layer=True)
    tensor = tf.nn.sigmoid(tensor)
    tensor = original * tensor
  return tensor


def resnet_block(tensor, filters, strides, training, weight_decay=0.0002,
                 kernel_size=3, activation=tf.nn.relu, drop_rate=0.0,
                 se_ratio=None):
  """Residual block."""
  original = tensor

  with tf.variable_scope("input"):
    tensor = batch_normalization(tensor, training=training)
    tensor = activation(tensor)
    tensor = tf.layers.conv2d(
      tensor, filters, kernel_size, strides, padding="same", use_bias=False,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

  with tf.variable_scope("output"):
    tensor = tf.layers.dropout(tensor, drop_rate)
    tensor = batch_normalization(tensor, training=training)
    tensor = activation(tensor)
    tensor = tf.layers.conv2d(
      tensor, filters, kernel_size, padding="same", use_bias=False,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    if se_ratio is not None:
      tensor = squeeze_and_excite(tensor, se_ratio)

    in_dims = original.shape[-1].value
    if in_dims != filters or strides > 1:
      diff = filters - in_dims
      original = tf.layers.average_pooling2d(original, strides, strides)
      original = tf.pad(original, [[0, 0], [0, 0], [0, 0], [0, diff]])

    tensor += original

  return tensor


def resnet_blocks(tensor, filters, strides, sub_layers, training, drop_rates,
                  **kwargs):
  if drop_rates is None:
    drop_rates = [0.] * len(filters)
  elif isinstance(drop_rates, numbers.Number):
    drop_rates = [drop_rates] * len(filters)

  for i, (filter, stride, drp) in enumerate(zip(filters, strides, drop_rates)):
    with tf.variable_scope("group_%d" % i):
      for j in range(sub_layers):
        with tf.variable_scope("block_%d" % j):
          stride = stride if j == 0 else 1
          tensor = resnet_block(
            tensor, filter, stride, training, drop_rate=drp, **kwargs)
  return tensor



# Hooks
class ExamplesPerSecondHook(tf.train.SessionRunHook):
  """Hook to print out examples per second."""

  def __init__(
      self,
      batch_size,
      every_n_iter=100,
      every_n_secs=None):
    """Initializer for ExamplesPerSecondHook."""
    if (every_n_iter is None) == (every_n_secs is None):
      raise ValueError("exactly one of every_n_steps"
                       " and every_n_secs should be provided.")
    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_iter, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use StepCounterHook.")

  def before_run(self, run_context):
    del run_context
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    del run_context

    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        current_examples_per_sec = steps_per_sec * self._batch_size
        tf.logging.info("Examples/sec: %g (%g), step = %g",
                     average_examples_per_sec, current_examples_per_sec,
                     self._total_steps)


class LoggingTensorHook(tf.train.SessionRunHook):
  """Hook to print batch of tensors."""

  def __init__(self, collection, every_n_iter=None, every_n_secs=None,
               batch=False, first_k=3):
    """Initializes a `LoggingTensorHook`."""
    self._collection = collection
    self._batch = batch
    self._first_k = first_k
    self._timer = tf.train.SecondOrStepTimer(
      every_secs=every_n_secs, every_steps=every_n_iter)

  def begin(self):
    self._timer.reset()
    self._iter_count = 0

  def before_run(self, run_context):  # pylint: disable=unused-argument
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      tensors = {t.name: t for t in tf.get_collection(self._collection)}
      return tf.train.SessionRunArgs(tensors)
    else:
      return None

  def _log_tensors(self, tensor_values):
    elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
    if self._batch:
      self._batch_print(tensor_values)
    else:
      self._print(tensor_values)

  def after_run(self, run_context, run_values):
    _ = run_context
    if self._should_trigger:
      self._log_tensors(run_values.results)

    self._iter_count += 1

  def _print(self, tensor_values):
    if not tensor_values:
      return
    for k, v in tensor_values.items():
      tf.logging.info("%s: %s", k, np.array_str(v))

  def _batch_print(self, tensor_values):
    if not tensor_values:
      return
    batch_size = list(tensor_values.values())[0].shape[0]
    for i in range(min(self._first_k, batch_size)):
      for k, v in tensor_values.items():
        tf.logging.info("{0}: {1}".format(k, v[i]))


class SummarySaverHook(tf.train.SessionRunHook):
  """Saves summaries every N steps."""

  def __init__(self,
               every_n_iter=None,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):
    """Initializes a `SummarySaverHook`."""
    self._summary_writer = summary_writer
    self._output_dir = output_dir
    self._timer = tf.train.SecondOrStepTimer(
      every_secs=every_n_iter, every_steps=every_n_secs)

  def begin(self):
    if self._summary_writer is None and self._output_dir:
      self._summary_writer = tf.summary.FileWriterCache.get(self._output_dir)
    self._next_step = None
    self._global_step_tensor = tf.train.get_global_step()
    self._summaries = tf.summary.merge_all()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use SummarySaverHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    self._request_summary = (
        self._next_step is None or
        self._timer.should_trigger_for_step(self._next_step))
    requests = {"global_step": self._global_step_tensor}
    if self._request_summary and self._summaries is not None:
      requests["summary"] = self._summaries

    return tf.train.SessionRunArgs(requests)

  def after_run(self, run_context, run_values):
    _ = run_context
    if not self._summary_writer:
      return

    global_step = run_values.results["global_step"]

    if self._request_summary:
      self._timer.update_last_triggered_step(global_step)
      if "summary" in run_values.results:
        summary = run_values.results["summary"]
        self._summary_writer.add_summary(summary, global_step)

    self._next_step = global_step + 1

  def end(self, session=None):
    if self._summary_writer:
      self._summary_writer.flush()