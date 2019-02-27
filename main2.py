
import numpy as np
import tensorflow as tf
from util import  load_data3
import argparse
import os
import util



class Model:
    def __init__(self):
        (self.train_images, self.train_labels,
         self.valid_images, self.valid_labels,
         self.all_images, self.all_labels) = load_data3()

    def get_model(self):
        tf.reset_default_graph()
        drop_rate = 0.5

        # specify the network
        x = tf.placeholder(tf.float32, [None, 32,32,3], name='input_placeholder')
        norm  = tf.divide(x, tf.constant(255, tf.float32), name='norm')

        features = util.conv_layers(norm,
                                   filters=[64, 192, 384, 256, 256],
                                   kernels=[3, 3, 3, 3, 3],
                                   pool_sizes=[2, 2, 2, 2, 2])
        features = tf.contrib.layers.flatten(features)

        output = util.dense_layers(
                    features, [512, 100],
                    drop_rates=drop_rate,
                    linear_top_layer=True)

        output = tf.identity(output, name='output')
        
        # define classification loss
        y = tf.placeholder(tf.int32, name='label')
        total = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=output)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = total + 0.1 * sum(reg_losses)
        return (output, total_loss, x, y)

    def evaluate(self, test_images, test_labels, confusion_matrix_op, total_loss, output, session,
                 x, y, batch_size, test_num_examples):
        ce_vals = []
        conf_mxs = []

        c_pred = tf.cast(tf.equal(tf.argmax(y, axis=1), output), tf.int32)
        c_preds = []
        for i in range(test_num_examples // batch_size):
            batch_xs = test_images[i * batch_size:(i + 1) * batch_size]
            batch_ys = test_labels[i * batch_size:(i + 1) * batch_size]
            test_ce, conf_matrix, c_pred_val= session.run(
                [tf.reduce_mean(total_loss), confusion_matrix_op, c_pred], {
                    x: batch_xs,
                    y: batch_ys
                })
            ce_vals.append(test_ce)
            conf_mxs.append(conf_matrix)
            c_preds += c_pred_val.tolist()
        return (ce_vals, conf_mxs, sum(c_preds)/len(c_preds))
    
    
    def init_model_base_params(self):
        self.dropout_factor = 0.3
        self.reg_constant = 0.01
        self.activation = tf.nn.relu
        self.episode = 100
        self.learning_rate = 0.0001
        self.batch_size = 128
    
    def update_model_info(self, model_id="9999999"):
        self.model_path = "homework_2" + str(model_id)
        os.makedirs(self.model_path, exist_ok=True)
        self.init_model_base_params()

    def train(self, model_id=0):
        self.update_model_info(model_id)
        train_num_examples = self.train_images.shape[0]
        valid_num_examples = self.valid_images.shape[0]

        output, total_loss, x, y = self.get_model()

        confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), 
                                                  tf.argmax(output, axis=1), num_classes=10)

        # set up training and saving functionality
        global_step_tensor = tf.get_variable(
            'global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
        saver = tf.train.Saver(max_to_keep=100)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            # run training
            batch_size = self.batch_size
            ce_vals = []
            print(train_num_examples // batch_size)
            for j in range(self.episode):
                for i in range(train_num_examples // batch_size):
                    batch_xs = self.train_images[i * batch_size:(i + 1) * batch_size]
                    batch_ys = self.train_labels[i * batch_size:(i + 1) * batch_size]
                    _, train_ce = session.run(
                        [train_op, tf.reduce_mean(total_loss)], {
                            x: batch_xs,
                            y: batch_ys
                        })
                    ce_vals.append(train_ce)
                    avg_train_ce = sum(ce_vals) / len(ce_vals)

                    if i % 50 == 0 and i > 10:
                        print('epoch:', j, ',step:', i, ', TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
                        # report mean validation loss
                        ce_vals, conf_mxs, acc = self.evaluate(self.valid_images, self.valid_labels, confusion_matrix_op, 
                                                                    total_loss, output, session,
                                                                    x, y, batch_size, valid_num_examples)
                        avg_test_ce = sum(ce_vals) / len(ce_vals)
                        print('VALIDATION CROSS ENTROPY: ' + str(avg_test_ce))
                        print('VALIDATION Accuracy     : ' + str(acc))

                        print(','.join(['TRAIN_PROGRESS', str(model_id), str(j), str(i), str(avg_train_ce), 
                                        str(avg_test_ce), str(acc) ]))


                if j % 3 == 0 and j > 1:
                    saver.save(
                            session,
                            os.path.join(self.model_path, "homework_2_" + str(model_id)) ,
                            global_step=global_step_tensor)
                    saver.save(
                            session,
                            os.path.join(self.model_path, "homework_2"))
                    ce_vals, conf_mxs, acc = self.evaluate(self.valid_images, self.valid_labels, confusion_matrix_op, 
                                        total_loss, output, session,
                                        x, y, batch_size, test_num_examples)
                    avg_test_ce = sum(ce_vals) / len(ce_vals)
                    print('------------------------------')
                    print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
                    print('TEST Accuracy     : ' + str(acc))

                    ce_vals2, conf_mxs2, acc2 = self.evaluate(self.all_images, self.all_labels, confusion_matrix_op, 
                                        total_loss, output, session,
                                        x, y, batch_size, test_num_examples)
                    avg_test_ce2 = sum(ce_vals2) / len(ce_vals2)
                    print('------------------------------')
                    print('GENERAL CROSS ENTROPY: ' + str(avg_test_ce2))
                    print('GENERAL Accuracy     : ' + str(acc2))


                    print(','.join(['TEST_PROGRESS', str(model_id), str(j), str(i), str(avg_train_ce), 
                                        str(avg_test_ce), str(acc),
                                        str(avg_test_ce2), str(acc2) ]))


                     


            # report mean test loss
            ce_vals, conf_mxs, acc = self.evaluate(self.valid_images, self.valid_labels, confusion_matrix_op, 
                                                   total_loss, output, session,
                                                   x, y, batch_size, test_num_examples)
            avg_test_ce = sum(ce_vals) / len(ce_vals)
            print('------------------------------')
            print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
            print('TEST Accuracy     : ' + str(acc))
            print('TEST CONFUSION MATRIX:')
            print(str(sum(conf_mxs)))
            ce_vals2, conf_mxs2, acc2 = self.evaluate(self.all_images, self.all_labels, confusion_matrix_op, 
                                total_loss, output, session,
                                x, y, batch_size, test_num_examples)
            avg_test_ce2 = sum(ce_vals2) / len(ce_vals2)
            print('------------------------------')
            print('GENERAL CROSS ENTROPY: ' + str(avg_test_ce2))
            print('GENERAL Accuracy     : ' + str(acc2))
            print('GENERAL CONFUSION MATRIX:')
            print(str(sum(conf_mxs2)))

            print(','.join(['TEST_PROGRESS', str(model_id), str(j), str(i), str(avg_train_ce), 
                                str(avg_test_ce), str(acc),
                                str(avg_test_ce2), str(acc2) ]))

            saver.save(
                session,
                os.path.join(self.model_path, "homework_2_" + str(model_id)) ,
                global_step=global_step_tensor)
            saver.save(
                session,
                os.path.join(self.model_path, "homework_2"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify MNIST images.')
    parser.add_argument(
        '--model_id',
        type=str,
        default="00000",
        help='rum the model id')
    args = parser.parse_args()

    m = Model()
    m.train(model_id=args.model_id)