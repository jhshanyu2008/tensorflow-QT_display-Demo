import numpy as np
import random
import tensorflow as tf
import os
import os.path as osp

root_dir = osp.dirname(__file__)
record_dir = osp.join(root_dir, 'data')

DATA_FILE = "credit_card.txt"

label_dic = {0: [0, 1],
             1: [1, 0]}


# Generate training and test sets from TXT file
def load_data(data_file=DATA_FILE):
    train_list = []
    test_list = []
    with open(data_file) as f:
        iter = 0
        f_data_train = []
        t_data_train = []
        f_data_test = []
        t_data_test = []
        # form raw training and test lists
        for sample in f.readlines():
            iter += 1
            # map function does a same implementation for every member of list
            values = list(map(float, sample.strip().split()))
            if iter <= 400:
                f_data_train.append(values)
            elif iter <= 492:
                f_data_test.append(values)
            elif iter <= 4492:
                t_data_train.append(values)
            else:
                t_data_test.append(values)
    # Balance the number of right and fault samples
    for i in range(len(t_data_train) // len(f_data_train)):
        train_list.extend(f_data_train)
    train_list.extend(t_data_train)
    random.shuffle(train_list)
    train_array = np.array(train_list)

    # Balance the number of right and fault samples
    for i in range(len(t_data_test) // len(f_data_test)):
        test_list.extend(f_data_test)
    test_list.extend(t_data_test)
    test_array = np.array(test_list)
    return train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]


# Merge the samples with their labels and form .tfrecords file
def create_data_record(data_array, label_array, tf_file):
    if not osp.exists(record_dir):
        os.makedirs(record_dir)
    writer = tf.python_io.TFRecordWriter(tf_file)
    length = len(label_array)
    for i in range(length):
        data = data_array[i, :]
        # extend it to 3-dim data which seems like a picture
        data = data[np.newaxis, :, np.newaxis]
        data_raw = data.tobytes()
        data_label = int(label_array[i])
        # merge sample and its label into an example
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "data_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_label])),
                    "data_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw]))
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()


# Form training .tfrecord file
def create_training_record(train_array, label_array, tf_file='data/train_credit.tfrecords'):
    create_data_record(train_array, label_array, tf_file)


# Form test .tfrecord file
def create_test_record(test_array, label_array, tf_file='data/test_credit.tfrecords'):
    create_data_record(test_array, label_array, tf_file)


# Read .tfrecord file and decode its content
def read_data_record(file_name):
    # read a series of files, here is only one member anyway
    filename_queue = tf.train.string_input_producer([file_name], shuffle=False, num_epochs=None)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    # read content from serialized_example
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data_label': tf.FixedLenFeature([], tf.int64),
            'data_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    # get content by name
    label_raw = features['data_label']
    data_raw = features['data_raw']
    # decode and reshape
    data = tf.decode_raw(data_raw, tf.float64)
    data = tf.reshape(data, [1 * 28 * 1])
    return data, label_raw


# Read training .tfrecord file
def read_training_record(file_name='data/train_credit.tfrecords'):
    return read_data_record(file_name)


# Read test .tfrecord file
def read_test_record(file_name='data/test_credit.tfrecords'):
    return read_data_record(file_name)


if __name__ == '__main__':
    # These codes will automatically generate training and test .tfrecord files
    print('creating tf records')
    train_data, train_labels, test_data, test_labels = load_data()
    create_training_record(train_data, train_labels)
    create_test_record(test_data, test_labels)
    print('create mission completed')

sess = tf.InteractiveSession()
train_sample, train_label = read_training_record()
# Read the training data from the queue, batch size is 5
train_sample_batch, train_label_batch = tf.train.batch([train_sample, train_label], batch_size=5)
test_sample, test_label = read_test_record()
# Read the test data from the queue, batch size is 180
test_sample_batch, test_label_batch = tf.train.batch([test_sample, test_label], batch_size=180)
# Run the queue
threads = tf.train.start_queue_runners(sess=sess)
init = tf.local_variables_initializer()
sess.run(init)


# Get training samples from queue
def get_training_data():
    data, labels = sess.run([train_sample_batch, train_label_batch])
    new_labes = []
    for i in range(len(labels)):
        new_labes.append(label_dic[labels[i]])
    return data, np.array(new_labes)


# Get test samples from queue
def get_test_data():
    data, labels = sess.run([test_sample_batch, test_label_batch])
    new_labes = []
    for i in range(len(labels)):
        new_labes.append(label_dic[labels[i]])
    return data, np.array(new_labes)
