import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import time

def shuffle_data(data, label):
    """Shuffle permutation of data."""
    num = data.shape[0]
    p = np.random.permutation(num)
    return (data[p,:], label[p,:])


def batch_generator(data, label, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects .
    """
    
    if shuffle:
        data, label = shuffle_data(data, label)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data.shape[0]:
            batch_count = 0

            if shuffle:
                data, label = shuffle_data(data, label)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield (data[start:end,:], label[start:end,:])


class MCD_DA():
    def __init__(self):
        pass

    def weight_variable(self, name, shape):
        initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01, dtype = tf.float32)
        return tf.get_variable(name, shape, initializer = initializer)

    def bias_variable(self, name, shape):
        initializer = tf.constant_initializer(value = 0.0, dtype = tf.float32)
        return tf.get_variable(name, shape, initializer = initializer)

    def alpha_variable(self, name):
        initializer = tf.constant_initializer(value = 0.75, dtype = tf.float32)
        return tf.get_variable(name, shape = (), initializer = initializer)

    def generator(self, x, filter_size, n_filters_1, n_filters_2, n_filters_3, n_units, keep_prob, reuse = False):
        x_reshaped = tf.reshape(x, [-1, 32, 32, 1])

        with tf.variable_scope('generator', reuse = reuse):
            w_1 = self.weight_variable('w_1', [filter_size, filter_size, 1, n_filters_1])
            b_1 = self.bias_variable('b_1', [n_filters_1])

            # conv1
            conv1 = tf.nn.conv2d(x_reshaped, w_1, strides = [1, 1, 1, 1], padding = 'SAME') +b_1

            # batch norm 1
            batch_mean_1, batch_var_1 = tf.nn.moments(conv1, [0, 1, 2])
            conv1 = (conv1 - batch_mean_1) / (tf.sqrt(batch_var_1) + 1e-5)

            # relu
            conv1 = tf.nn.relu(conv1)

            # max_pool_1
            conv1 = tf.nn.max_pool(conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

            
            w_2 = self.weight_variable('w_2', [filter_size, filter_size, n_filters_1, n_filters_2])
            b_2 = self.bias_variable('b_2', [n_filters_2])

            # conv2
            conv2 = tf.nn.conv2d(conv1, w_2, strides = [1, 1, 1, 1], padding = 'SAME') + b_2

            # batch norm 2
            batch_mean_2, batch_var_2 = tf.nn.moments(conv2, [0, 1, 2])
            conv2 = (conv2 - batch_mean_2) / (tf.sqrt(batch_var_2) + 1e-5)
            
            # relu 2
            conv2 = tf.nn.relu(conv2)
            
            # max_pool_2
            conv2 = tf.nn.max_pool(conv2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

            w_3 = self.weight_variable('w_3', [filter_size, filter_size, n_filters_2, n_filters_3])
            b_3 = self.bias_variable('b_3', [n_filters_3])
            
            # conv3
            conv3 = tf.nn.conv2d(conv2, w_3, strides = [1, 1, 1, 1], padding = 'SAME') + b_3

            # batch norm 3
            batch_mean_3, batch_var_3 = tf.nn.moments(conv3, [0, 1, 2])
            conv3 = (conv3 - batch_mean_3) / (tf.sqrt(batch_var_3) + 1e-5)
            
            # relu 3
            conv3 = tf.nn.relu(conv3)

                        
            
            # fc 1
            conv_flat = tf.reshape(conv3, [-1, 8 * 8 * n_filters_3])   # 8192=8*8*128

            w_4 = self.weight_variable('w_4', [8 * 8 * n_filters_3, n_units])
            b_4 = self.bias_variable('b_4', [n_units])
            
            fc = tf.matmul(conv_flat, w_4) + b_4

            # bn1_fc
            batch_mean, batch_var = tf.nn.moments(fc, [0])
            bn1_fc = (fc - batch_mean) / (tf.sqrt(batch_var) + 1e-5)

            # relu
            fc = tf.nn.relu(bn1_fc)

            # dropout
            fc = tf.nn.dropout(fc, keep_prob)
            
                        
            # leaky relu
            #fc = tf.maximum(0.2 * fc, fc)
            
            feature = fc
        return feature

    def classifier_1(self, 
                     x, 
                     n_units_1, 
                     n_units_2, 
                     #keep_prob, 
                     reuse = False):
        
        with tf.variable_scope('classifier_1', reuse = reuse):
            w_1 = self.weight_variable('w_1', [n_units_1, n_units_2])
            b_1 = self.bias_variable('b_1', [n_units_2])
            
            fc = tf.matmul(x, w_1) + b_1

            # batch norm
            batch_mean, batch_var = tf.nn.moments(fc, [0])
            fc = (fc - batch_mean) / (tf.sqrt(batch_var) + 1e-5)
            
            # relu
            fc = tf.nn.relu(fc)
            
            w_2 = self.weight_variable('w_2', [n_units_2, 10])
            b_2 = self.bias_variable('b_2', [10])
            
            fc = tf.matmul(fc, w_2) + b_2
            
            #dropout
            #fc = tf.nn.dropout(fc, keep_prob)

            logits = fc
        return logits

    def classifier_2(self, 
                     x, 
                     n_units_1, 
                     n_units_2, 
                     #keep_prob, 
                     reuse = False):
        
        with tf.variable_scope('classifier_2', reuse = reuse):
            
            w_1 = self.weight_variable('w_1', [n_units_1, n_units_2])
            b_1 = self.bias_variable('b_1', [n_units_2])
            
            fc = tf.matmul(x, w_1) + b_1

            # batch norm
            batch_mean, batch_var = tf.nn.moments(fc, [0])
            fc = (fc - batch_mean) / (tf.sqrt(batch_var) + 1e-5)
            
            # relu
            fc = tf.nn.relu(fc)
            
            w_2 = self.weight_variable('w_2', [n_units_2, 10])
            b_2 = self.bias_variable('b_2', [10])
            
            fc = tf.matmul(fc, w_2) + b_2
            
            #dropout
            #fc = tf.nn.dropout(fc, keep_prob)

            logits = fc
        return logits

    def loss_cross_entropy(self, logits, labels):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return cross_entropy

    def discrepancy(self, probs_1, probs_2):
        return tf.reduce_mean(tf.abs(probs_1 - probs_2)) 

    def accuracy(self, y, t):
        correct_preds = tf.equal(tf.argmax(y, axis = 1), tf.argmax(t, axis = 1))
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        return accuracy

    def training(self, loss, learning_rate, var_list):
        #optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)
        train_step = optimizer.minimize(loss, var_list = var_list)
        return train_step

    def training_clipped(self, loss, learning_rate, clip_norm, var_list):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

        grads_and_vars = optimizer.compute_gradients(loss, var_list = var_list)
        clipped_grads_and_vars = [(tf.clip_by_norm(grad, clip_norm = clip_norm), \
                                 var) for grad, var in grads_and_vars]
        train_step = optimizer.apply_gradients(clipped_grads_and_vars)

        return train_step

    def train(self, images_train_1, labels_train_1, images_test_1, labels_test_1, \
            images_train_t, labels_train_t, images_test_t, labels_test_t, \
            filter_size, n_filters_1, n_filters_2, n_filters_3, n_units_g, n_units_c, \
            learning_rate, epoch, num_stepC, batch_size, show_step, is_saving, model_path):
        
        tf.reset_default_graph()

        x_1 = tf.placeholder(shape = [None, 32 * 32], dtype = tf.float32)
        y_1 = tf.placeholder(shape = [None, 10], dtype = tf.float32)
        x_t = tf.placeholder(shape = [None, 32 * 32], dtype = tf.float32)
        y_t = tf.placeholder(shape = [None, 10], dtype = tf.float32)
        keep_prob = tf.placeholder(shape = (), dtype = tf.float32)

        feat_1 = self.generator(x_1, filter_size, n_filters_1, n_filters_2, n_filters_3,  n_units_g, \
                                keep_prob, reuse = False)
        feat_t = self.generator(x_t, filter_size, n_filters_1, n_filters_2, n_filters_3, n_units_g, \
                                keep_prob, reuse = True)

        logits_1_1 = self.classifier_1(feat_1, n_units_g, n_units_c, reuse = False)
        probs_1_1 = tf.nn.softmax(logits_1_1)
        loss_1_1 = self.loss_cross_entropy(logits_1_1, y_1)

        logits_1_t = self.classifier_1(feat_t, n_units_g, n_units_c, reuse = True)
        probs_1_t = tf.nn.softmax(logits_1_t)
        loss_1_t = self.loss_cross_entropy(logits_1_t, y_t)

        logits_2_1 = self.classifier_2(feat_1, n_units_g, n_units_c, reuse = False)
        probs_2_1 = tf.nn.softmax(logits_2_1)
        loss_2_1 = self.loss_cross_entropy(logits_2_1, y_1)

        logits_2_t = self.classifier_2(feat_t, n_units_g, n_units_c, reuse = True)
        probs_2_t = tf.nn.softmax(logits_2_t)
        loss_2_t = self.loss_cross_entropy(logits_2_t, y_t)

        loss_a = loss_1_1 + loss_2_1
        #loss_b = - self.discrepancy(probs_1_t, probs_2_t)
        loss_b = loss_a - self.discrepancy(probs_1_t, probs_2_t)
        loss_c = self.discrepancy(probs_1_t, probs_2_t)

        var_list_g = tf.trainable_variables('generator')
        var_list_c_1 = tf.trainable_variables('classifier_1')
        var_list_c_2 = tf.trainable_variables('classifier_2')

        var_list_a = var_list_g + var_list_c_1 + var_list_c_2
        var_list_b = var_list_c_1 + var_list_c_2
        var_list_c = var_list_g

        # Without Gradient Clipping
        train_step_a = self.training(loss_a, learning_rate, var_list_a)
        train_step_b = self.training(loss_b, learning_rate, var_list_b)
        train_step_c = self.training(loss_c, learning_rate, var_list_c)

        acc_1_1 =  self.accuracy(probs_1_1, y_1)
        acc_1_t =  self.accuracy(probs_1_t, y_t)
        acc_2_1 =  self.accuracy(probs_2_1, y_1)
        acc_2_t =  self.accuracy(probs_2_t, y_t)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            history_loss_train_1_1 = []
            history_loss_train_1_t = []
            history_loss_train_2_1 = []
            history_loss_train_2_t = []

            history_loss_test_1_1 = []
            history_loss_test_1_t = []
            history_loss_test_2_1 = []
            history_loss_test_2_t = []

            history_acc_train_1_1 = []
            history_acc_train_1_t = []
            history_acc_train_2_1 = []
            history_acc_train_2_t = []

            history_acc_test_1_1 = []
            history_acc_test_1_t = []
            history_acc_test_2_1 = []
            history_acc_test_2_t = []

            history_loss_train_a = []
            history_loss_train_b = []
            history_loss_train_c = []

            history_loss_test_a = []
            history_loss_test_b = []
            history_loss_test_c = []
            
            get_source_batch = batch_generator(images_train_1, labels_train_1, batch_size, shuffle=True)
            get_target_batch = batch_generator(images_train_t, labels_train_t, batch_size, shuffle=True)
            
            test_source_batch = batch_generator(images_test_1, labels_test_1, batch_size, shuffle=False)
            test_target_batch = batch_generator(images_test_t, labels_test_t, batch_size, shuffle=False)
            
            
            #n_iter = images_train_1.shape[0]//batch_size
            n_iter = images_train_t.shape[0]//batch_size
            print('number of batches for training: {}'.format(n_iter))
            
            iter_total = -1
            best_acc_classifier1 = 0
            best_acc_classifier2 = 0
            show_numbers = 0
            #cur_model_name = 'MCD_net_{}'.format(int(time.time()))
            
            # variables for plot
            source_acc_train1 = []
            source_acc_train2 = []
            target_acc_train1 = []
            target_acc_train2 = []
            
            source_acc_test1 = []
            source_acc_test2 = []
            target_acc_test1 = []
            target_acc_test2 = []
            
            for epc in range(epoch):
                print("epoch {} ".format(epc + 1))

                for i in range(n_iter):
                    iter_total += 1
                    
                    x_train_batch, y_train_batch = next(get_source_batch)
                    x_target_batch, y_target_batch = next(get_target_batch)
                    
                    # Train
                    # Step A
                    feed_A_dict = {x_1: x_train_batch, y_1: y_train_batch, keep_prob: 0.5} #0.8
                    sess.run(train_step_a, feed_dict = feed_A_dict)

                    # Step B
                    feed_B_dict = {x_1: x_train_batch, y_1: y_train_batch, x_t: x_target_batch, keep_prob: 0.5} #0.8
                    sess.run(train_step_b, feed_dict = feed_B_dict)

                    # Step C
                    for n_c in range(num_stepC):
                        feed_C_dict = {x_t: x_target_batch, keep_prob: 0.5} #0.75
                        sess.run(train_step_c, feed_dict = feed_C_dict) 
                        

                    # Checking
                    """
                    # Train data
                    rand_index = np.random.choice(len(images_train_1), size = batch_size)
                    x_batch_1 = images_train_1[rand_index]
                    y_batch_1 = labels_train_1[rand_index]

                    rand_index = np.random.choice(len(images_train_t), size = batch_size)
                    x_batch_t = images_train_t[rand_index]
                    y_batch_t = labels_train_t[rand_index]

                    feed_dict = {x_1: x_batch_1, y_1: y_batch_1, x_t: x_batch_t, y_t: y_batch_t, keep_prob: 0.8} #0.8

                    temp_loss_train_1_1 = sess.run(loss_1_1, feed_dict = feed_dict)
                    temp_loss_train_1_t = sess.run(loss_1_t, feed_dict = feed_dict)
                    temp_loss_train_2_1 = sess.run(loss_2_1, feed_dict = feed_dict)
                    temp_loss_train_2_t = sess.run(loss_2_t, feed_dict = feed_dict)

                    temp_acc_train_1_1 = sess.run(acc_1_1, feed_dict = feed_dict)
                    temp_acc_train_1_t = sess.run(acc_1_t, feed_dict = feed_dict)
                    temp_acc_train_2_1 = sess.run(acc_2_1, feed_dict = feed_dict)
                    temp_acc_train_2_t = sess.run(acc_2_t, feed_dict = feed_dict)

                    history_loss_train_1_1.append(temp_loss_train_1_1)
                    history_loss_train_1_t.append(temp_loss_train_1_t)
                    history_loss_train_2_1.append(temp_loss_train_2_1)
                    history_loss_train_2_t.append(temp_loss_train_2_t)

                    history_acc_train_1_1.append(temp_acc_train_1_1)
                    history_acc_train_1_t.append(temp_acc_train_1_t)
                    history_acc_train_2_1.append(temp_acc_train_2_1)
                    history_acc_train_2_t.append(temp_acc_train_2_t)

                    temp_loss_train_a = sess.run(loss_a, feed_dict = feed_dict)
                    temp_loss_train_b = sess.run(loss_b, feed_dict = feed_dict)
                    temp_loss_train_c = sess.run(loss_c, feed_dict = feed_dict)

                    history_loss_train_a.append(temp_loss_train_a)
                    history_loss_train_b.append(temp_loss_train_b)
                    history_loss_train_c.append(temp_loss_train_c)

                    # Test data
                    rand_index = np.random.choice(len(images_test_1), size = batch_size)
                    x_batch_1 = images_test_1[rand_index]
                    y_batch_1 = labels_test_1[rand_index]

                    rand_index = np.random.choice(len(images_test_t), size = batch_size)
                    x_batch_t = images_test_t[rand_index]
                    y_batch_t = labels_test_t[rand_index]

                    feed_dict = {x_1: x_batch_1, y_1: y_batch_1, x_t: x_batch_t, y_t: y_batch_t, keep_prob: 0.8} #0.8

                    temp_loss_test_1_1 = sess.run(loss_1_1, feed_dict = feed_dict)
                    temp_loss_test_1_t = sess.run(loss_1_t, feed_dict = feed_dict)
                    temp_loss_test_2_1 = sess.run(loss_2_1, feed_dict = feed_dict)
                    temp_loss_test_2_t = sess.run(loss_2_t, feed_dict = feed_dict)

                    temp_acc_test_1_1 = sess.run(acc_1_1, feed_dict = feed_dict)
                    temp_acc_test_1_t = sess.run(acc_1_t, feed_dict = feed_dict)
                    temp_acc_test_2_1 = sess.run(acc_2_1, feed_dict = feed_dict)
                    temp_acc_test_2_t = sess.run(acc_2_t, feed_dict = feed_dict)

                    history_loss_test_1_1.append(temp_loss_test_1_1)
                    history_loss_test_1_t.append(temp_loss_test_1_t)
                    history_loss_test_2_1.append(temp_loss_test_2_1)
                    history_loss_test_2_t.append(temp_loss_test_2_t)

                    history_acc_test_1_1.append(temp_acc_test_1_1)
                    history_acc_test_1_t.append(temp_acc_test_1_t)
                    history_acc_test_2_1.append(temp_acc_test_2_1)
                    history_acc_test_2_t.append(temp_acc_test_2_t)

                    temp_loss_test_a = sess.run(loss_a, feed_dict = feed_dict)
                    temp_loss_test_b = sess.run(loss_b, feed_dict = feed_dict)
                    temp_loss_test_c = sess.run(loss_c, feed_dict = feed_dict)

                    history_loss_test_a.append(temp_loss_test_a)
                    history_loss_test_b.append(temp_loss_test_b)
                    history_loss_test_c.append(temp_loss_test_c)
                    """
                    
                    if i % show_step == 0:
                        show_numbers += 1
                        print ('-' * 15)
                        # do validation
                        # Accurancy
                        acc_source_test1 = 0
                        acc_source_test2 = 0
                        acc_target_test1 = 0
                        acc_target_test2 = 0
                        
                        acc_source_train1 = 0
                        acc_source_train2 = 0
                        acc_target_train1 = 0
                        acc_target_train2 = 0
                        # Loss
                        loss_test_a = 0
                        loss_test_b = 0
                        loss_test_c = 0
                        
                        for acc_n in range(n_iter):
                            #test
            
                            x_train_batch, y_train_batch = next(test_source_batch)
                            x_target_batch, y_target_batch = next(test_target_batch)
                            
                            feed_test_dict = {x_1: x_train_batch, y_1: y_train_batch, 
                                         x_t: x_target_batch, y_t: y_target_batch, keep_prob: 0.8} #0.8
                            
                            acc_source_test1 += sess.run(acc_1_1, feed_dict=feed_test_dict)
                            acc_source_test2 += sess.run(acc_2_1, feed_dict=feed_test_dict)
                            acc_target_test1 += sess.run(acc_1_t, feed_dict=feed_test_dict)
                            acc_target_test2 += sess.run(acc_2_t, feed_dict=feed_test_dict)
                            
                            loss_test_a += sess.run(loss_a, feed_dict = feed_test_dict)
                            loss_test_b += sess.run(loss_b, feed_dict = feed_test_dict)
                            loss_test_c += sess.run(loss_c, feed_dict = feed_test_dict)
                            
                            #train
                            x_train_batch, y_train_batch = next(get_source_batch)
                            x_target_batch, y_target_batch = next(get_target_batch)
                            
                            feed_test_dict = {x_1: x_train_batch, y_1: y_train_batch, 
                                         x_t: x_target_batch, y_t: y_target_batch, keep_prob: 0.5} #0.8
                            
                            acc_source_train1 += sess.run(acc_1_1, feed_dict=feed_test_dict)
                            acc_source_train2 += sess.run(acc_2_1, feed_dict=feed_test_dict)
                            acc_target_train1 += sess.run(acc_1_t, feed_dict=feed_test_dict)
                            acc_target_train2 += sess.run(acc_2_t, feed_dict=feed_test_dict)
                     
                        acc_source_test1 /= n_iter
                        acc_source_test2 /= n_iter
                        acc_target_test1 /= n_iter
                        acc_target_test2 /= n_iter
                        
                        acc_source_train1 /= n_iter
                        acc_source_train2 /= n_iter
                        acc_target_train1 /= n_iter
                        acc_target_train2 /= n_iter
                        
                        source_acc_test1.append(acc_source_test1) 
                        source_acc_test2.append(acc_source_test2)
                        target_acc_test1.append(acc_target_test1)
                        target_acc_test2.append(acc_target_test2)
                        
                        source_acc_train1.append(acc_source_train1)
                        source_acc_train2.append(acc_source_train2)
                        target_acc_train1.append(acc_target_train1)
                        target_acc_train2.append(acc_target_train2)


                        loss_test_a /= n_iter
                        loss_test_b /= n_iter
                        loss_test_c /= n_iter
                            
                            
                        """
                        if verbose:
                            print('{}/{} loss: {} validation accuracy : {}%'.format(
                                batch_size * (i + 1),
                                X_train.shape[0],
                                cur_loss,
                                valid_acc))
                        
                        # save the merge result summary
                        writer.add_summary(merge_result, iter_total)
                        """
                        print("Iteration: {}, Loss_test_a: {}, Loss_test_b: {}, Loss_test_c: {}".format(iter_total, 
                                                                                                        loss_test_a, 
                                                                                                        loss_test_b,
                                                                                                        loss_test_c))
                        print("Source test acc1: {}%, Source test acc2: {}%.".format(100*acc_source_test1,
                                                                                       100*acc_source_test2))


                        # when achieve the best validation accuracy, we store the model paramters
                        
                        if acc_target_test1 > best_acc_classifier1:
                            print('Best test accuracy classifier1! Iteration:{} accuracy: {}%'.format(iter_total,
                                                                                                   100*acc_target_test1))
                            best_acc_classifier1 = acc_target_test1
                        if acc_target_test2 > best_acc_classifier2:
                            print('Best test accuracy classifier2! Iteration:{} accuracy: {}%'.format(iter_total,
                                                                                                      100*acc_target_test2))
                            best_acc_classifier2 = acc_target_test2
                            
                            #saver.save(sess, 'model/{}'.format(cur_model_name))

            
            #print("Traning ends. The best test accuracy is classifier1: {}, classifier2: {}. Model named {}.".format(best_acc_classifier1, best_acc_classifier1, cur_model_name))
            print("Traning ends. The best test accuracy is classifier1: {}%, classifier2: {}%.".format(100*best_acc_classifier1, 100*best_acc_classifier2))

            print ('-'* 15)    
            fig = plt.figure(figsize = (10, 3))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(range(show_numbers), source_acc_train1, 'b-', label = 'Source train acc')
            ax1.plot(range(show_numbers), source_acc_test1, 'r--', label = 'Source test acc')
            ax1.set_title('Classifier1 Acc for Source Data')
            ax1.legend(loc = 'lower right')

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(range(show_numbers), target_acc_train1, 'b-', label = 'Target train acc')
            ax2.plot(range(show_numbers), target_acc_test1, 'r--', label = 'Target test acc')
            ax2.set_ylim(0.0, 1.0)
            ax2.set_title('Classifier1 Acc for Target Data')
            ax2.legend(loc = 'lower right')

            plt.show()

            print ('-'* 15)    
            fig = plt.figure(figsize = (10, 3))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(range(show_numbers), source_acc_train2, 'b-', label = 'Source train acc')
            ax1.plot(range(show_numbers), source_acc_test2, 'r--', label = 'Source test acc')
            ax1.set_title('Classifier2 Acc for Source Data')
            ax1.legend(loc = 'lower right')

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(range(show_numbers), target_acc_train2, 'b-', label = 'Target train acc')
            ax2.plot(range(show_numbers), target_acc_test2, 'r--', label = 'Target test acc')
            ax2.set_ylim(0.0, 1.0)
            ax2.set_title('Classifier2 Acc for Target Data')
            ax2.legend(loc = 'lower right')

            plt.show()

           