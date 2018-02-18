import sys
import scipy.io.wavfile
import pandas
import os
import zipfile
import numpy as np
import tensorflow as tf

batch_size = 2

width = 1000
height = 1764
num_channels = 2
num_labels = 2

patch_size = 5
depth = 40

num_hidden = 500
# data shape = (1764000, 2)
def real_get_data():
    path = '/home/williamjlee/data_clean'
    rand_nongun = np.random.choice(os.listdir(os.path.join(path, 'nongun')), batch_size / 2, replace=False)
    rand_gun = np.random.choice(os.listdir(os.path.join(path, 'gun')), batch_size / 2, replace=False)
    
    nongun_test = np.random.choice(os.listdir(os.path.join(path, 'nongun_test')), batch_size / 2, replace=False)
    gun_test = np.random.choice(os.listdir(os.path.join(path, 'gun_test')), batch_size / 2, replace=False)
    
    train = []
    leng = 352800
    for f in rand_gun:
        #r, data = scipy.io.wavfile.read(os.path.join(path, 'gun', f))
        with open(os.path.join(path, 'gun', f), 'rb') as rp:
            r, data = pickle.load(rp)
        result = np.zeros((leng, 2))
        m = min(data.shape[0], leng)
        result[:m, :2] = data[:m, :2]
        result = np.reshape(result, (1000, -1, result.shape[1]))
        train.append(result)
    for f in rand_nongun:
        #r, data = scipy.io.wavfile.read(os.path.join(path, 'nongun', f))
        with open(os.path.join(path, 'nongun', f), 'rb') as rp:
            r, data = pickle.load(rp)
        result = np.zeros((leng, 2))
        m = min(data.shape[0], leng)
        result[:m, :2] = data[:m, :2]
        result = np.reshape(result, (1000, -1, result.shape[1]))
        train.append(result)
    train = np.array(train)
    train_label = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],
                      [0,1],[0,1],[0,1],[0,1],[0,1]])
    
    test = []
    for f in gun_test:
        #r, data = scipy.io.wavfile.read(os.path.join(path, 'gun_test', f))
        with open(os.path.join(path, 'gun_test', f), 'rb') as rp:
            r, data = pickle.load(rp)
        result = np.zeros((leng, 2))
        m = min(data.shape[0], leng)
        result[:m, :2] = data[:m, :2]
        result = np.reshape(result, (1000, -1, result.shape[1]))
        test.append(result)
    for f in nongun_test:
        #r, data = scipy.io.wavfile.read(os.path.join(path, 'nongun_test', f))
        with open(os.path.join(path, 'nongun_test', f), 'rb') as rp:
            r, data = pickle.load(rp)
        result = np.zeros((leng, 2))
        m = min(data.shape[0], leng)
        result[:m, :2] = data[:m, :2]
        result = np.reshape(result, (1000, -1, result.shape[1]))
        test.append(result)
    test = np.array(test)
    test_label = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],
                      [0,1],[0,1],[0,1],[0,1],[0,1]])
    yield train, train_label, test, test_label


# 5 of each class
num_iter = 1000
num_report = 10
def train_loop(train, test, train_labels, test_labels, runs):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    feed_dict = dict()
    for step in range(num_iter):
        x, y, tx, ty = real_get_data()
        feed_dict[train] = x
        feed_dict[train_labels] = y
        feed_dict[test] = tx
        feed_dict[test_labels] = ty
        res = sess.run(runs, feed_dict=feed_dict)
        print(step)
        if step != 0 and step % 10 == 0:
            print('Opt: {}'.format(res[0]))
            print('Loss: {}'.format(res[1]))
            print('Pred: {}'.format(res[2]))




def model_run(train=True):
    tf.reset_default_graph()
    print('hi')
    test_dataset = tf.placeholder(tf.float32, shape=(batch_size, width, None, num_channels))
    test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    train_dataset = tf.placeholder(tf.float32, shape=(batch_size, width, None, num_channels))
    train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    
    #conv1
    stddev1 = np.sqrt(2.0 / (patch_size * patch_size * num_channels))
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=stddev1
    ))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    
    # conv2
    stddev2 = np.sqrt(2.0 / (patch_size * patch_size * depth))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=stddev2
    ))
    layer2_biases = tf.Variable(tf.zeros([depth]))
    
    # l3
    stretch_size = width // 4 * height // 4 * depth
    stddev3 = np.sqrt(2.0 / stretch_size)
    layer3_weights = tf.Variable(tf.truncated_normal(
        [stretch_size, num_hidden], stddev=stddev3
    ))
    layer3_biases = tf.Variable(tf.zeros(
        [num_hidden]
    ))
    
    # l4
    stddev4 = np.sqrt(2.0 / num_hidden)
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=stddev4
    ))
    layer4_biases = tf.Variable(tf.zeros([num_labels]))
    
    def model(data):
        layer1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(layer1 + layer1_biases)
        pool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        layer2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(layer2 + layer2_biases)
        pool2 = tf.nn.max_pool(relu2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], -1])
        layer3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

        layer4 = tf.matmul(layer3, layer4_weights) + layer4_biases
        return layer4
    
    logits = model(train_dataset)
    train_prediction = tf.nn.softmax(logits)
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits)
        )
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    test_prediction = tf.nn.softmax(model(test_dataset))
    
    if train:
        train_loop(train_dataset, test_dataset, train_labels, test_labels, [optimizer, loss, train_prediction])

print('starting')
model_run(train=True)



