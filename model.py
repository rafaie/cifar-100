"""
model.py: it includes the base model and other models
"""

import tensorflow as tf
import numpy as np
import os
import functools
# import hooks
import util

tf.logging.set_verbosity(tf.logging.INFO)  # if you want to see the log info

class BaseConfig(object):
    def __init__(self, name='test_model'):
        self.name = name
        self.learning_rate = 0.001

        self.model_dir = './' + self.name
        os.makedirs(self.model_dir, exist_ok=True)

        # config for `tf.data.Dataset`
        self.shuffle_and_repeat = True
        self.shuffle_buffer_size = 10000
        self.batch_size = 128
        self.decay_rate = 0.1
        self.decay_steps = 10000
        self.constant_steps = 20000


        # training configuration
        self.keep_checkpoint_max = 5
        self.save_checkpoints_steps = 400
        self.stop_if_no_increase_hook_max_steps_without_increase = 5000
        self.stop_if_no_increase_hook_min_steps = 50000
        self.stop_if_no_increase_hook_run_every_secs = 120
        self.save_summary_steps = 400
        self.num_epochs = 20000
        self.throttle_secs = 200
        self.wit_hook = True


class BaseModel(object):
    def __init__(self, params=None):
        self.config = None
        self.init_config(params)
        print(params)

    def load_dataset(self, dataset, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.config.shuffle_and_repeat is True:
                dataset = dataset.shuffle(self.config.shuffle_buffer_size).repeat(
                                          self.config.num_epochs).batch(self.config.batch_size)
            else:
                dataset = dataset.batch(self.config.batch_size)
        elif mode == tf.estimator.ModeKeys.EVAL:
            dataset = dataset.batch(self.config.batch_size)

        ds_iter = dataset.make_one_shot_iterator()
        return ds_iter.get_next()

    def train_and_evaluate(self, ds_train, ds_eval):
        # Prepare dataset
        it_train = functools.partial(self.load_dataset, ds_train, tf.estimator.ModeKeys.TRAIN)
        it_eval = functools.partial(self.load_dataset, ds_eval, tf.estimator.ModeKeys.EVAL)

        # Session Cconfiguration
        session_config = tf.ConfigProto()
        session_config.allow_soft_placement = True
        session_config.gpu_options.allow_growth = True


        cfg = tf.estimator.RunConfig(model_dir=self.config.model_dir,
                                     save_summary_steps=self.config.save_summary_steps,
                                     save_checkpoints_steps=self.config.save_checkpoints_steps,
                                     save_checkpoints_secs=None,
                                     session_config=session_config,
                                     keep_checkpoint_max=self.config.keep_checkpoint_max)

        estimator = tf.estimator.Estimator(model_fn = self._model_fn, 
                                           config=cfg)

        train_hooks, eval_hooks = self.get_hooks(estimator)
        train_spec = tf.estimator.TrainSpec(input_fn=it_train, hooks=train_hooks)
        eval_spec = tf.estimator.EvalSpec(input_fn=it_eval, hooks=eval_hooks, throttle_secs=self.config.throttle_secs)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    def get_hooks(self, estimator):
        train_hooks =[ 
                    util.ExamplesPerSecondHook(
                        batch_size=self.config.batch_size,
                        every_n_iter=self.config.save_summary_steps),
                    util.LoggingTensorHook(
                        collection="batch_logging",
                        every_n_iter=self.config.save_summary_steps,
                        batch=True),
                    util.LoggingTensorHook(
                        collection="logging",
                        every_n_iter=self.config.save_summary_steps,
                        batch=False),
                    tf.contrib.estimator.stop_if_no_increase_hook(
                        estimator, "accuracy", 
                        max_steps_without_increase=self.config.stop_if_no_increase_hook_max_steps_without_increase, 
                        min_steps = self.config.stop_if_no_increase_hook_min_steps)]
        
        eval_hooks = [
                    util.SummarySaverHook(every_n_iter=self.config.save_summary_steps,
                                           output_dir=os.path.join(self.config.model_dir, "eval"))]

        return (train_hooks, eval_hooks)

    def get_learning_rate(self):
        return self.config.learning_rate

    def _model_fn(self, features, labels, mode, params={}):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = self.get_learning_rate()
        opt = tf.train.AdamOptimizer(learning_rate)

        predictions, loss, eval_metrics = self.model_fn(features, labels, mode, params)
        losses = [loss]
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        loss = None

        if losses:
            loss = tf.add_n(losses) / len(losses)
            tf.summary.scalar("loss/main", tf.add_n(losses))

        if reg_losses:
            loss += tf.add_n(reg_losses)
            tf.summary.scalar("loss/regularization", tf.add_n(reg_losses))

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.control_dependencies(update_ops):
                train_op = opt.minimize(loss, 
                                        global_step=global_step,
                                        colocate_gradients_with_ops=True)

            opts = tf.profiler.ProfileOptionBuilder().trainable_variables_parameter()
            stats = tf.profiler.profile(tf.get_default_graph(), options=opts)
            print("Total parameters:", stats.total_parameters)
        else:
            train_op = None

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops=eval_metrics)

    def init_config(self, params=None):
        raise NotImplementedError

    def model_fn(self, features, labels, mode, params={}):
        raise NotImplementedError



class ModelCNN(BaseModel):
    def init_config(self, params=None):
        self.config = BaseConfig('ModelCNN')
        self.config.weight_decay = 0.0002
        self.config.drop_rate = 0.3
        self.config.normalization_val = 1


    
    def model_fn(self, images, labels, mode, params):
        """CNN classifier model."""
        # images = features["image"]
        # labels = labels["label"]

        training = mode == tf.estimator.ModeKeys.TRAIN
        drop_rate = self.config.drop_rate if training else 0.0

        # features = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_placeholder')
        features = tf.divide(images, tf.constant(self.config.normalization_val, tf.float32), name='input_placeholder')
        features = util.conv_layers(features, [16], [3], linear_top_layer=True,
                                    weight_decay=self.config.weight_decay)

        features = util.resnet_blocks(
                    features, [16, 32, 64], [1, 2, 2], 5, training=training,
                    weight_decay=self.config.weight_decay,
                    drop_rates=drop_rate)

        features = util.batch_normalization(features, training=training)
        features = tf.nn.relu(features)

        logits = util.dense_layers(features, [100],
                                    linear_top_layer=False,
                                    weight_decay=self.config.weight_decay)
        logits = tf.reduce_mean(logits, axis=[1, 2])
        logits = tf.identity(logits, name='model_output')

        predictions = tf.argmax(logits, axis=-1)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        eval_metrics = {
            "accuracy": tf.metrics.accuracy(labels, predictions),
            "top_1_error": tf.metrics.mean(util.top_k_error(labels, logits, 1)),
        }

        return {"predictions": predictions}, loss, eval_metrics

class AlexNet(BaseModel):
    def init_config(self, params=None):
        self.config = BaseConfig('AlexNet')
        self.config.weight_decay = 0.002
        self.config.drop_rate = 0.5
        self.config.normalization_val = 1


    
    def model_fn(self, images, labels, mode, params):
        """CNN classifier model."""

        training = mode == tf.estimator.ModeKeys.TRAIN
        drop_rate = self.config.drop_rate if training else 0.0

        features = tf.divide(images, tf.constant(self.config.normalization_val, tf.float32), name='input_placeholder')

        # features = util.conv_layers(
        #                             images,
        #                             filters=[64, 192, 384, 256, 256],
        #                             kernels=[11, 5, 3, 3, 3],
        #                             strides=[4, 1, 1, 1],
        #                             pool_sizes=[2, 2, 2, 2, 2],
        #                             pool_strides=[2, 2, 2, 2, 2]
        #                            )

        features = util.conv_layers(images,
                                   filters=[64, 128, 256],
                                   kernels=[3, 3, 3],
                                   pool_sizes=[2, 2, 2])
        features = tf.contrib.layers.flatten(features)

        logits = util.dense_layers(
                    features, [512, 100],
                    drop_rates=drop_rate,
                    linear_top_layer=True)

        predictions = tf.argmax(logits, axis=-1)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        logits = tf.identity(logits, name='model_output')

        predictions = tf.argmax(logits, axis=-1)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        tf.summary.image("images", images)

        eval_metrics = {
            "accuracy": tf.metrics.accuracy(labels, predictions),
            "top_1_error": tf.metrics.mean(util.top_k_error(labels, logits, 1)),
        }

        tf.add_to_collection(
            "batch_logging", tf.identity(labels, name="labels"))
        tf.add_to_collection(
            "batch_logging", tf.identity(predictions, name="predictions"))

        return {"predictions": predictions}, loss, eval_metrics
