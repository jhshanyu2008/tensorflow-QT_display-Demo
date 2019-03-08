import tensorflow as tf
from Qt_display.Qt_matplotlib import Basic_Canvas
from data_load_credit import get_training_data, get_test_data
import os
import os.path as osp
from Qt_display.display_window_FC import Func_MainWindow
from PyQt5 import QtWidgets
from multiprocessing import Process, Queue
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QVBoxLayout
import sys

root_dir = osp.dirname(__file__)
model_dir = osp.join(root_dir, 'models')

print_data_q = Queue()
analysis_q = Queue()


def Main_Process(print_q, analy_q):
    """
    :param print_q: A queue using for storing the data needs to be printed
    :param analy_q: A queue using for storing analysis data
    """
    def Print(message):
        print(message)
        print_q.put(message)

    x = tf.placeholder(tf.float32, shape=[None, 1 * 28 * 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])

    # ---------------VGG Convolution Neural Network--------------- #
    # input——>7 convolutional layers——>3 densely layers——>softmax
    # ---------------paraments preprocessing--------------- #
    # W parameter preprocessing
    def weight_variable(shape):
        # stddev is the standard deviation
        initial = tf.truncated_normal(shape, stddev=0.1)
        # Returns the four-dimensional tensor by truncate processing
        return tf.Variable(initial)

    # The initial value of all of the b parameters is 0.1
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # ---------------Convolution and Pooling functions--------------- #
    def conv2d(x1, W_1):
        # input_data = x1 :input，4-dim tensor [batch, in_height, in_width, in_deep]
        # filter = W_1 :kernel，4-dim tensor [filter_height, filter_width, in_channels, out_channels]
        # strides :The slide size of the kernel，4-dim [1, stride,stride, 1]
        # padding :The SAME mode will expands the boundary if the data field is not enough
        # use_cudnn_on_gpu whether use gpu or not
        # return a 4-dim tensor
        return tf.nn.conv2d(x1, W_1, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_1x2(x1):
        # value :x1 input
        # ksize ：pooling size，4-dim [batch, height, width, deep] batch deep always 1
        # strides :The slide size of pooling [1, stride,stride, 1]
        # padding :The SAME mode will expands the boundary if the data field is not enough
        # return a 4-dim tensor
        return tf.nn.max_pool(x1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    # ---------------Input Layer--------------- #
    input_data = tf.reshape(x, [-1, 1, 28, 1])

    # ---------------First Convolutional Layers--------------- #
    # Two layers，input 1*28*1,output 1*14*64
    w_conv1_1 = weight_variable([1, 3, 1, 64])
    b_conv1_1 = bias_variable([64])

    h_cov1_1 = tf.nn.relu(conv2d(input_data, w_conv1_1) + b_conv1_1)

    w_conv1_2 = weight_variable([1, 3, 64, 64])
    b_conv1_2 = bias_variable([64])

    h_cov1_2 = tf.nn.relu(conv2d(h_cov1_1, w_conv1_2) + b_conv1_2)
    h_pool_1 = max_pool_1x2(h_cov1_2)

    # ---------------Second Convolutional Layers--------------- #
    # Two layers，input 1*14*64,output 1*7*128
    w_conv2_1 = weight_variable([1, 3, 64, 128])
    b_conv2_1 = bias_variable([128])

    h_cov2_1 = tf.nn.relu(conv2d(h_pool_1, w_conv2_1) + b_conv2_1)

    w_conv2_2 = weight_variable([1, 3, 128, 128])
    b_conv2_2 = bias_variable([128])

    h_cov2_2 = tf.nn.relu(conv2d(h_cov2_1, w_conv2_2) + b_conv2_2)
    h_pool_2 = max_pool_1x2(h_cov2_2)

    # ---------------Third Convolutional Layers--------------- #
    # Three layers，input 1*7*128,output 1*4*256
    w_conv3_1 = weight_variable([1, 3, 128, 256])
    b_conv3_1 = bias_variable([256])

    h_cov3_1 = tf.nn.relu(conv2d(h_pool_2, w_conv3_1) + b_conv3_1)

    w_conv3_2 = weight_variable([1, 3, 256, 256])
    b_conv3_2 = bias_variable([256])

    h_cov3_2 = tf.nn.relu(conv2d(h_cov3_1, w_conv3_2) + b_conv3_2)

    w_conv3_3 = weight_variable([1, 3, 256, 256])
    b_conv3_3 = bias_variable([256])

    h_cov3_3 = tf.nn.relu(conv2d(h_cov3_2, w_conv3_3) + b_conv3_3)
    h_pool_3 = max_pool_1x2(h_cov3_3)

    # ---------------First Densely Connected Layer--------------- #
    # One layer，input 1*4*256，output 1024
    W_fc1 = weight_variable([1 * 4 * 256, 1024])
    b_fc1 = bias_variable([1024])

    # reshape h_pool_3
    h_pool5_flat = tf.reshape(h_pool_3, [-1, 1 * 4 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    # drop_out process
    keep_prob_1 = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_1)

    # ---------------second Densely Connected Layer--------------- #
    # One layer，input 1024，output 1024
    W_fc2 = weight_variable([1024, 1024])
    b_fc2 = bias_variable([1024])

    h_fc2_flat = tf.reshape(h_fc1_drop, [-1, 1024])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc2_flat, W_fc2) + b_fc2)

    keep_prob_2 = tf.placeholder(tf.float32)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_2)

    # ---------------Third Densely Connected Layer--------------- #
    # one layer，input 1024，output 500
    W_fc3 = weight_variable([1024, 500])
    b_fc3 = bias_variable([500])

    h_fc3_flat = tf.reshape(h_fc2_drop, [-1, 1024])
    h_fc3 = tf.nn.relu(tf.matmul(h_fc3_flat, W_fc3) + b_fc3)

    keep_prob_3 = tf.placeholder(tf.float32)
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob_3)

    # ---------------Output Layer--------------- #
    # 500 input，2 output
    W_soft = weight_variable([500, 2])
    b_soft = bias_variable([2])

    # prediction used for softmax
    y_predict = tf.matmul(h_fc3_drop, W_soft) + b_soft

    # ---------------Train and Evaluate--------------- #
    # define cross entropy and main training step
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # define accuracy
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10000)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # flag determines training or just test
    Train = True
    if Train:
        # ——————The commented context are used for display by matplotlib using its normal way——————
        # process for visualization
        # plt.ion()
        # plt.figure()
        # ax1 = plt.subplot(211)
        # ax1.set_title("Cross Entropy")
        # ax1.set_ylabel('Cross entropy')
        # ax2 = plt.subplot(212)
        # ax2.set_title("Train Accuracy")
        # ax2.set_xlabel('Iteration')
        # ax2.set_ylabel('Train Accuracy')
        # use to store the test accuracy
        test_accuracy = []
        test_iter = []
        # ————if need pretrained model,uncomment the next line,and change the model name which you need————
        # saver.restore(sess, "models/credit/credit_model120000.ckpt")
        for i in range(20000):
            batch = get_training_data()
            if (i + 1) % 100 == 0:
                # check training accuracy and cross entropy then plot and print them
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],
                                                          keep_prob_1: 0.5, keep_prob_2: 1, keep_prob_3: 1})
                entropy = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1],
                                                        keep_prob_1: 0.5, keep_prob_2: 1, keep_prob_3: 1})
                analy_q.put([i, entropy, train_accuracy])
                # ax1.bar(i, entropy, width=80, facecolor="#9999ff", edgecolor="white")
                # ax2.bar(i, train_accuracy, width=80, facecolor="#ff9999", edgecolor="white")
                # plt.draw()
                # plt.pause(1)
                Print("step {0}, training accuracy: {1}".format(i + 1, train_accuracy))
                Print("cross entropy: {}\n".format(entropy))
            # run the main training process
            train_step.run(feed_dict={x: batch[0], y_: batch[1],
                                      keep_prob_1: 0.5, keep_prob_2: 1, keep_prob_3: 1})
            # check training result by every 200 times'training
            if (i + 1) % 200 == 0:
                rate = 0
                # 180 * 10 = 1800  batch size of test is set as 180 (see data_load_credit.py line 135)
                # And the number of test samples
                for j in range(10):
                    batch = get_test_data()
                    rate += accuracy.eval(feed_dict={x: batch[0], y_: batch[1],
                                                     keep_prob_1: 0.5, keep_prob_2: 1, keep_prob_3: 1})
                mean_accuracy = 100 * rate / 10
                # store test accuracy
                test_iter.append(i + 1)
                test_accuracy.append(mean_accuracy)
                Print('The total accuracy is {0}%'.format(mean_accuracy))
                # save models
                if mean_accuracy > 95:
                    if not osp.exists(model_dir):
                        os.makedirs(model_dir)
                    saver_path = "{0}/credit_model{1}.ckpt".format(model_dir, i + 1)
                    saver.save(sess, saver_path)
                    Print("model saved in file:{0}".format(saver_path))
        # plt.savefig('{0}/{1}.jpg'.format('data', 'result_credit'), dpi=300)
        # plt.ioff()
        # plt.figure()
        # ax = plt.subplot(111)
        # ax.set_title("Test Accuracy")
        # ax.set_xlabel('Iteration')
        # ax.set_ylabel('Test Accuracy')
        # # plot the change of test accuracy
        # ax.bar(test_iter, test_accuracy, width=180, facecolor="#9999ff", edgecolor="white")
        # plt.savefig('{0}/{1}.jpg'.format('data', 'result2_credit'), dpi=100)
        # plt.show()
    if not Train:
        # Don't forget to change the model name which you need
        saver.restore(sess, "models/credit/credit_model12000.ckpt")
    rate = 0
    for i in range(10):
        batch = get_test_data()
        rate += accuracy.eval(feed_dict={x: batch[0], y_: batch[1],
                                         keep_prob_1: 0.5, keep_prob_2: 1, keep_prob_3: 1})
    Print('The total accuracy is {0}%'.format(100 * rate / 10))


class Matplot_Canvas(Basic_Canvas):

    def __init__(self, *args, **kwargs):
        super(Matplot_Canvas, self).__init__(*args, **kwargs)
        self.analy_q = analysis_q
        self.ax1.hold(True)
        self.ax2.hold(True)
        self.ax1.set_title("Cross Entropy")
        self.ax1.set_ylabel('Cross entropy')
        self.ax2.set_title("Train Accuracy")
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Train Accuracy')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)

    def update_plot(self):
        while True:
            if self.analy_q.empty():
                break
            else:
                i, entropy, train_accuracy = self.analy_q.get()
                self.ax1.bar(i, entropy, width=80, facecolor="#9999ff", edgecolor="white")
                self.ax2.bar(i, train_accuracy, width=80, facecolor="#ff9999", edgecolor="white")
        self.draw()


class Qt_display(Func_MainWindow):
    def __init__(self):
        super(Qt_display, self).__init__()
        self.queue = print_data_q
        self.get_data_timer = QTimer()  # init a timer
        self.get_data_timer.timeout.connect(self.get_queue_data)
        self.get_data_timer.start(50)  # 设置计时间隔并启动
        self.setupUi(self)  # init UI, this line is inevitable

    def setupUi(self, MainWindow):
        super(Qt_display, self).setupUi(MainWindow)
        self.setWindowTitle("Monitor")   # set title
        # The details of the design for plt_layer
        self.plt_layer = QVBoxLayout(self.centralwidget)
        self.plt_layer.setContentsMargins(340, 10, 10, 10)
        self.plt_layer.setObjectName("pltLayer")
        # add the Matplot_QT context
        matplot_plt = Matplot_Canvas(width=5, height=4, dpi=100)
        self.plt_layer.addWidget(matplot_plt)

    def get_queue_data(self):
        while True:
            if self.queue.empty():
                break
            else:
                self.send_to_display(self.queue.get())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Main_window = Qt_display()
    Main_window.show()
    Process_thread = Process(target=Main_Process, args=(print_data_q, analysis_q,))
    Process_thread.start()
    sys.exit(app.exec_())
