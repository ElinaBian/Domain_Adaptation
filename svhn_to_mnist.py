# Maximum Classifier Discrepancy Domain Adaptation
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

def weight_variable(name, shape):
    initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01, dtype = tf.float32)
    return tf.get_variable(name, shape, initializer = initializer)

def bias_variable(name, shape):
    initializer = tf.constant_initializer(value = 0.0, dtype = tf.float32)
    return tf.get_variable(name, shape, initializer = initializer)

def alpha_variable(name):
    initializer = tf.constant_initializer(value = 0.75, dtype = tf.float32)
    return tf.get_variable(name, shape = (), initializer = initializer)

def generator(x, filter_size, n_filters_1, n_filters_2, n_filters_3, n_units, keep_prob, reuse = False):
    x_reshaped = tf.reshape(x, [-1, 32, 32, 1])

    with tf.variable_scope('generator', reuse = reuse):
        w_1 = weight_variable('w_1', [filter_size, filter_size, 1, n_filters_1])
        b_1 = bias_variable('b_1', [n_filters_1])

        # conv1
        conv1 = tf.nn.conv2d(x_reshaped, w_1, strides = [1, 1, 1, 1], padding = 'SAME') +b_1

        # batch norm 1
        batch_mean_1, batch_var_1 = tf.nn.moments(conv1, [0, 1, 2])
        conv1 = (conv1 - batch_mean_1) / (tf.sqrt(batch_var_1) + 1e-5)

        # relu
        conv1 = tf.nn.relu(conv1)

        # max_pool_1
        conv1 = tf.nn.max_pool(conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')


        w_2 = weight_variable('w_2', [filter_size, filter_size, n_filters_1, n_filters_2])
        b_2 = bias_variable('b_2', [n_filters_2])

        # conv2
        conv2 = tf.nn.conv2d(conv1, w_2, strides = [1, 1, 1, 1], padding = 'SAME') + b_2

        # batch norm 2
        batch_mean_2, batch_var_2 = tf.nn.moments(conv2, [0, 1, 2])
        conv2 = (conv2 - batch_mean_2) / (tf.sqrt(batch_var_2) + 1e-5)

        # relu 2
        conv2 = tf.nn.relu(conv2)

        # max_pool_2
        conv2 = tf.nn.max_pool(conv2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        w_3 = weight_variable('w_3', [filter_size, filter_size, n_filters_2, n_filters_3])
        b_3 = bias_variable('b_3', [n_filters_3])

        # conv3
        conv3 = tf.nn.conv2d(conv2, w_3, strides = [1, 1, 1, 1], padding = 'SAME') + b_3

        # batch norm 3
        batch_mean_3, batch_var_3 = tf.nn.moments(conv3, [0, 1, 2])
        conv3 = (conv3 - batch_mean_3) / (tf.sqrt(batch_var_3) + 1e-5)

        # relu 3
        conv3 = tf.nn.relu(conv3)



        # fc 1
        conv_flat = tf.reshape(conv3, [-1, 8 * 8 * n_filters_3])   # 8192=8*8*128

        w_4 = weight_variable('w_4', [8 * 8 * n_filters_3, n_units])
        b_4 = bias_variable('b_4', [n_units])

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

def classifier_1(x, n_units_1, n_units_2, reuse = False):

    with tf.variable_scope('classifier_1', reuse = reuse):
        w_1 = weight_variable('w_1', [n_units_1, n_units_2])
        b_1 = bias_variable('b_1', [n_units_2])

        fc = tf.matmul(x, w_1) + b_1

        # batch norm
        batch_mean, batch_var = tf.nn.moments(fc, [0])
        fc = (fc - batch_mean) / (tf.sqrt(batch_var) + 1e-5)

        # relu
        fc = tf.nn.relu(fc)

        w_2 = weight_variable('w_2', [n_units_2, 10])
        b_2 = bias_variable('b_2', [10])

        fc = tf.matmul(fc, w_2) + b_2

        #dropout
        #fc = tf.nn.dropout(fc, keep_prob)

        logits = fc
    return logits

def classifier_2(x, n_units_1, n_units_2, reuse = False):

    with tf.variable_scope('classifier_2', reuse = reuse):

        w_1 = weight_variable('w_1', [n_units_1, n_units_2])
        b_1 = bias_variable('b_1', [n_units_2])

        fc = tf.matmul(x, w_1) + b_1

        # batch norm
        batch_mean, batch_var = tf.nn.moments(fc, [0])
        fc = (fc - batch_mean) / (tf.sqrt(batch_var) + 1e-5)

        # relu
        fc = tf.nn.relu(fc)

        w_2 = weight_variable('w_2', [n_units_2, 10])
        b_2 = bias_variable('b_2', [10])

        fc = tf.matmul(fc, w_2) + b_2

        #dropout
        #fc = tf.nn.dropout(fc, keep_prob)

        logits = fc
    return logits

def loss_cross_entropy(y, t):
    cross_entropy = - tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis = 1))
    return cross_entropy

def loss_discriminator(probs_1, probs_2):
    return - tf.reduce_mean(tf.reduce_sum(tf.abs(probs_1 - probs_2), axis = 1)) 

def loss_generator(probs_1, probs_2):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(probs_1 - probs_2), axis = 1)) 

def loss_entropy(p):
    entropy = - tf.reduce_mean(tf.reduce_sum(p * tf.log(tf.clip_by_value(p, 1e-10, 1.0)), axis = 1))
    return entropy

def loss_mutual_information(p):
    p_ave = tf.reduce_mean(p, axis = 0)
    h_y = -tf.reduce_sum(p_ave * tf.log(p_ave + 1e-16))
    h_y_x = - tf.reduce_mean(tf.reduce_sum(p * tf.log(tf.clip_by_value(p, 1e-10, 1.0)), axis = 1))
    mutual_info = h_y - h_y_x
    return -mutual_info

def accuracy(y, t):
    correct_preds = tf.equal(tf.argmax(y, axis = 1), tf.argmax(t, axis = 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    return accuracy

def training(loss, learning_rate, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train_step = optimizer.minimize(loss, var_list = var_list)
    return train_step

def training_clipped(loss, learning_rate, clip_norm, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

    grads_and_vars = optimizer.compute_gradients(loss, var_list = var_list)
    clipped_grads_and_vars = [(tf.clip_by_norm(grad, clip_norm = clip_norm), \
                             var) for grad, var in grads_and_vars]
    train_step = optimizer.apply_gradients(clipped_grads_and_vars)

    return train_step

def fit(images_train_1, labels_train_1, images_test_1, labels_test_1, \
        images_train_t, labels_train_t, images_test_t, labels_test_t, \
        filter_size, n_filters_1, n_filters_2, n_filters_3, n_units_g, n_units_c, \
        learning_rate, n_iter, batch_size, show_step, is_saving, model_path):

    tf.reset_default_graph()

    x_1 = tf.placeholder(shape = [None, 32 * 32], dtype = tf.float32)
    y_1 = tf.placeholder(shape = [None, 10], dtype = tf.float32)
    x_t = tf.placeholder(shape = [None, 32 * 32], dtype = tf.float32)
    y_t = tf.placeholder(shape = [None, 10], dtype = tf.float32)
    keep_prob = tf.placeholder(shape = (), dtype = tf.float32)

    feat_1 = generator(x_1, filter_size, n_filters_1, n_filters_2, n_filters_3,  n_units_g, \
                            keep_prob, reuse = False)
    feat_t = generator(x_t, filter_size, n_filters_1, n_filters_2, n_filters_3, n_units_g, \
                            keep_prob, reuse = True)

    logits_1_1 = classifier_1(feat_1, n_units_g, n_units_c, reuse = False)
    probs_1_1 = tf.nn.softmax(logits_1_1)
    loss_1_1 = loss_cross_entropy(probs_1_1, y_1)

    logits_1_t = classifier_1(feat_t, n_units_g, n_units_c, reuse = True)
    probs_1_t = tf.nn.softmax(logits_1_t)
    loss_1_t = loss_cross_entropy(probs_1_t, y_t)

    logits_2_1 = classifier_2(feat_1, n_units_g, n_units_c, reuse = False)
    probs_2_1 = tf.nn.softmax(logits_2_1)
    loss_2_1 = loss_cross_entropy(probs_2_1, y_1)

    logits_2_t = classifier_2(feat_t, n_units_g, n_units_c, reuse = True)
    probs_2_t = tf.nn.softmax(logits_2_t)
    loss_2_t = loss_cross_entropy(probs_2_t, y_t)

    loss_a = loss_1_1 + loss_2_1
    loss_b = loss_a + loss_discriminator(probs_1_t, probs_2_t)
    #loss_b = loss_discriminator(probs_1_t, probs_2_t)
    loss_c = loss_generator(probs_1_t, probs_2_t)

    var_list_g = tf.trainable_variables('generator')
    var_list_c_1 = tf.trainable_variables('classifier_1')
    var_list_c_2 = tf.trainable_variables('classifier_2')

    var_list_a = var_list_g + var_list_c_1 + var_list_c_2
    var_list_b = var_list_c_1 + var_list_c_2
    var_list_c = var_list_g

    # Without Gradient Clipping
    train_step_a = training(loss_a, learning_rate, var_list_a)
    train_step_b = training(loss_b, learning_rate, var_list_b)
    train_step_c = training(loss_c, learning_rate, var_list_c)

    acc_1_1 =  accuracy(probs_1_1, y_1)
    acc_1_t =  accuracy(probs_1_t, y_t)
    acc_2_1 =  accuracy(probs_2_1, y_1)
    acc_2_t =  accuracy(probs_2_t, y_t)

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

        for i in range(n_iter):
            # Train
            # Step A
            rand_index = np.random.choice(len(images_train_1), size = batch_size)
            x_batch = images_train_1[rand_index]
            y_batch = labels_train_1[rand_index]

            feed_dict = {x_1: x_batch, y_1: y_batch, keep_prob: 0.8}

            sess.run(train_step_a, feed_dict = feed_dict)

            # Step B
            rand_index = np.random.choice(len(images_train_1), size = batch_size)
            x_batch_1 = images_train_1[rand_index]
            y_batch_1 = labels_train_1[rand_index]

            rand_index = np.random.choice(len(images_train_t), size = batch_size)
            x_batch_t = images_train_t[rand_index]

            feed_dict = {x_1: x_batch_1, y_1: y_batch_1, x_t: x_batch_t, keep_prob: 0.8}

            sess.run(train_step_b, feed_dict = feed_dict)

            # Step C
            rand_index = np.random.choice(len(images_train_t), size = batch_size)
            x_batch_t = images_train_t[rand_index]

            feed_dict = {x_t: x_batch_t, keep_prob: 0.75}

            sess.run(train_step_c, feed_dict = feed_dict) 

            # Checking
            # Train data
            rand_index = np.random.choice(len(images_train_1), size = batch_size)
            x_batch_1 = images_train_1[rand_index]
            y_batch_1 = labels_train_1[rand_index]

            rand_index = np.random.choice(len(images_train_t), size = batch_size)
            x_batch_t = images_train_t[rand_index]
            y_batch_t = labels_train_t[rand_index]

            feed_dict = {x_1: x_batch_1, y_1: y_batch_1, x_t: x_batch_t, y_t: y_batch_t, keep_prob: 0.8}

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

            if (i + 1) % show_step == 0:
                print ('-' * 15)
                print ('Iteration: ' + str(i + 1) + '  Loss_a: ' + str(temp_loss_train_a) + \
                       'Loss_b: ' + str(temp_loss_train_b) + '  Loss_c: ' + str(temp_loss_train_c))
                #print ('Iteration: ' + str(i + 1))

            # Test data
            rand_index = np.random.choice(len(images_test_1), size = batch_size)
            x_batch_1 = images_test_1[rand_index]
            y_batch_1 = labels_test_1[rand_index]

            rand_index = np.random.choice(len(images_test_t), size = batch_size)
            x_batch_t = images_test_t[rand_index]
            y_batch_t = labels_test_t[rand_index]

            feed_dict = {x_1: x_batch_1, y_1: y_batch_1, x_t: x_batch_t, y_t: y_batch_t, keep_prob: 0.8}

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

        print ('-'* 15)    
        fig = plt.figure(figsize = (10, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(range(n_iter), history_loss_train_1_1, 'b-', label = 'Train')
        ax1.plot(range(n_iter), history_loss_test_1_1, 'r--', label = 'Test')
        ax1.set_title('Loss_1_1')
        ax1.legend(loc = 'upper right')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(n_iter), history_acc_train_1_1, 'b-', label = 'Train')
        ax2.plot(range(n_iter), history_acc_test_1_1, 'r--', label = 'Test')
        ax2.set_ylim(0.0, 1.0)
        ax2.set_title('Accuracy_1_1')
        ax2.legend(loc = 'lower right')

        plt.show()

        print ('-'* 15)    
        fig = plt.figure(figsize = (10, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(range(n_iter), history_loss_train_2_1, 'b-', label = 'Train')
        ax1.plot(range(n_iter), history_loss_test_2_1, 'r--', label = 'Test')
        ax1.set_title('Loss_2_1')
        ax1.legend(loc = 'upper right')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(n_iter), history_acc_train_2_1, 'b-', label = 'Train')
        ax2.plot(range(n_iter), history_acc_test_2_1, 'r--', label = 'Test')
        ax2.set_ylim(0.0, 1.0)
        ax2.set_title('Accuracy_2_1')
        ax2.legend(loc = 'lower right')

        plt.show()

        print ('-'* 15)    
        fig = plt.figure(figsize = (10, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(range(n_iter), history_loss_train_1_t, 'b-', label = 'Train')
        ax1.plot(range(n_iter), history_loss_test_1_t, 'r--', label = 'Test')
        ax1.set_title('Loss_1_t')
        ax1.legend(loc = 'upper right')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(n_iter), history_acc_train_1_t, 'b-', label = 'Train')
        ax2.plot(range(n_iter), history_acc_test_1_t, 'r--', label = 'Test')
        ax2.set_ylim(0.0, 1.0)
        ax2.set_title('Accuracy_1_t')
        ax2.legend(loc = 'lower right')

        plt.show()

        print ('-'* 15)    
        fig = plt.figure(figsize = (10, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(range(n_iter), history_loss_train_2_t, 'b-', label = 'Train')
        ax1.plot(range(n_iter), history_loss_test_2_t, 'r--', label = 'Test')
        ax1.set_title('Loss_2_t')
        ax1.legend(loc = 'upper right')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(n_iter), history_acc_train_2_t, 'b-', label = 'Train')
        ax2.plot(range(n_iter), history_acc_test_2_t, 'r--', label = 'Test')
        ax2.set_ylim(0.0, 1.0)
        ax2.set_title('Accuracy_2_t')
        ax2.legend(loc = 'lower right')

        print ('-'* 15)    
        fig = plt.figure(figsize = (10, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(range(n_iter), history_loss_train_a, 'b-', label = 'Train_a')
        ax1.plot(range(n_iter), history_loss_train_b, 'r-', label = 'Train_b')
        ax1.plot(range(n_iter), history_loss_train_c, 'y-', label = 'Train_c')
        ax1.set_title('Loss_train')
        ax1.legend(loc = 'upper right')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(n_iter), history_loss_test_a, 'b-', label = 'Test_a')
        ax2.plot(range(n_iter), history_loss_test_b, 'r-', label = 'Test_b')
        ax2.plot(range(n_iter), history_loss_test_c, 'y-', label = 'Test_c')
        ax2.set_title('Loss_test')
        ax2.legend(loc = 'upper right')

        plt.show()

        if is_saving:
            model_path = saver.save(sess, model_path)
            print ('done saving at ', model_path)