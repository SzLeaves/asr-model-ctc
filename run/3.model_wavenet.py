#!/usr/bin/env python
# coding: utf-8

# 3. CTC + WaveNet模型
import os
import pathlib
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf

# # 设置显存大小
# gpus = tf.config.experimental.list_physical_devices("GPU")
# memory_size = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8704)
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [memory_size])

# 导入keras API
keras = tf.keras

ModelCheckpoint = keras.callbacks.ModelCheckpoint
ctc_batch_cost = keras.backend.ctc_batch_cost
Model = keras.models.Model
SGD = keras.optimizers.SGD
EarlyStopping, PiecewiseConstantDecay = (
    keras.callbacks.EarlyStopping,
    keras.optimizers.schedules.PiecewiseConstantDecay,
)
Input, BatchNormalization, Activation, Lambda = (
    keras.layers.Input,
    keras.layers.BatchNormalization,
    keras.layers.Activation,
    keras.layers.Lambda,
)
Conv1D, Multiply, Add = (
    keras.layers.Conv1D,
    keras.layers.Multiply,
    keras.layers.Add,
)

# 音频/语音标注文件路径
DS_PATH = "../data/"
# 模型文件路径
FILES_PATH = "../output/"


# 0. 读取数据集
# 读取音频特征
with open(FILES_PATH + "dataset/data_mfcc.pkl", "rb") as file:
    train_ds, mfcc_mean, mfcc_std = pickle.load(file)

# 读取音频标注
with open(FILES_PATH + "dataset/labels.pkl", "rb") as file:
    train_label = [x.strip().split() for x in pickle.load(file)]

# 读取词库
with open(FILES_PATH + "dataset/words_vec.pkl", "rb") as file:
    char2id, id2char = pickle.load(file)


# 1. 构建CTC模型的input / output


def ctc_loss(args):
    """
    构建CTC模型损失函数
    :param args: 输入ctc_batch_cost的参数
    :return:     (sample, 1) 每个批次内数据包含的CTC损失
    """
    y_true, y_pred, input_length, label_length = args
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)


def ctc_batch_generator(data, labels, dict_list, n_mfcc, max_length, batch_size):
    """
    构建模型输入使用的CTC模型格式数据, 包含长度对齐操作
    :param dict_list:
    :param data:        音频MFCC特征
    :param labels:      语音标注标签(已转换为数字)
    :param n_mfcc:      音频MFCC特征维数
    :param max_length:  标签最大填充长度
    :param batch_size:  每批次送入模型训练的数据数量
    :return:            (dict, dict) 包含符合CTC模型格式的input/output数据
    """
    # 初始批次数据量为0
    cur_batch = 0
    # 生成器
    while True:
        # 当前批次的数据量
        cur_batch += batch_size
        """
        这里使用 offset >= len(data) 判断条件是为了在offset索引超出长度时自动重置该值
        防止后续的list操作溢出 (X_data切片取不到batch_size长度的数值)
        同时重置offset值为最初的batch_size作为索引, 并重新打乱数据
        这样可以在之前所有批次数据取完后，重新给模型提供不一样的数据集(从头开始批次生成)
        """
        # 在加载每批次的数据前打乱排序
        if cur_batch == batch_size or cur_batch >= len(data):
            shuffle_index = np.arange(len(data))
            np.random.shuffle(shuffle_index)
            # 保证数据与标签一一对应，需要使用同一套已经打乱顺序的标签
            data = [data[x] for x in shuffle_index]
            labels = [labels[x] for x in shuffle_index]
            # 重置cur_batch索引值
            cur_batch = batch_size

        # 从数据集中获取一个批次的数据，个数为batch_size
        X_data = data[cur_batch - batch_size : cur_batch]
        y_data = labels[cur_batch - batch_size : cur_batch]

        # 获取音频最大帧数作为统一长度 (保证所有数据的完整性)
        max_frame = np.max([x.shape[0] for x in X_data])

        # 以下过程是先按最大长度创建空间, 然后将没有对齐的数据直接放入空间中, 达到整体对齐的目的（填充法）
        X_batch = np.zeros([batch_size, max_frame, n_mfcc])  # 输入的特征长度，填充为最大
        y_batch = np.ones([batch_size, max_length]) * len(dict_list)  # 输入的标签长度填充为总词数
        X_length = np.zeros([batch_size, 1], dtype=np.int16)  # 输入的数据量=批次数量
        y_length = np.zeros([batch_size, 1], dtype=np.int16)  # 输入的特征量=批次数量

        # 根据批次数据实时更新CTC输入
        for i in range(batch_size):
            X_length[i, 0] = X_data[i].shape[0]
            X_batch[i, : X_length[i, 0], :] = X_data[i]

            y_length[i, 0] = len(y_data[i])
            y_batch[i, : y_length[i, 0]] = [dict_list[x] for x in y_data[i]]

        # 保存构建的数据结构
        ctc_inputs = {
            "X": X_batch,
            "y": y_batch,
            "X_length": X_length,
            "y_length": y_length,
        }
        ctc_output = {"ctc": np.zeros([batch_size])}

        # generator 迭代数据
        yield ctc_inputs, ctc_output


# 2. 构建WaveNet模型


def model_wavenet(
    words_size,
    n_mfcc,
    filter_range,
    n_filters=128,
    n_blocks=3,
    kernel_size=7,
    learning_rate=0.02,
):
    """
    按照指定参数构建WaveNet模型
    :param words_size:     词库大小
    :param n_mfcc:         音频MFCC特征维数
    :param filter_range:   list 卷积核扩大间隔范围
    :param n_filters:      卷积核尺寸
    :param n_blocks:       扩大层数
    :param kernel_size:    扩大层(1-dim)卷积核尺寸
    :param learning_rate:  SGD优化器学习速率
    :return:               (wavenet, ctc_model) 返回构建的WaveNet模型和CTC Loss模型
    """

    # 一维卷积层
    def conv1d(inputs, filters, kernel_size, diltion_rate):
        return Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="causal",
            activation=None,
            dilation_rate=diltion_rate,
        )(inputs)

    # 标准化函数
    def normal(inputs):
        return BatchNormalization()(inputs)

    # 激活层函数
    def activation(inputs, activation):
        return Activation(activation)(inputs)

    # 扩大卷积网络
    def res_block(inputs, filters, kernel_size, dilation_rate):
        res_1 = activation(
            normal(conv1d(inputs, filters, kernel_size, dilation_rate)), "tanh"
        )
        res_2 = activation(
            normal(conv1d(inputs, filters, kernel_size, dilation_rate)), "sigmoid"
        )
        res_add = Multiply()([res_1, res_2])

        res_active_1 = activation(normal(conv1d(res_add, filters, 1, 1)), "tanh")
        res_active_2 = activation(normal(conv1d(res_add, filters, 1, 1)), "tanh")

        return Add()([res_active_1, inputs]), res_active_2

    # 定义模型输入数据格式 (输入格式与ctc_batch_generator的返回值一致)
    input_data = Input(shape=(None, n_mfcc), dtype=np.float32, name="X")

    # 定义WaveNet模型结构 #
    wav_1 = activation(normal(conv1d(input_data, n_filters, 1, 1)), "tanh")
    shortcut = []
    for index in range(n_blocks):
        for r in filter_range:
            wav_1, s = res_block(wav_1, n_filters, kernel_size, r)
            shortcut.append(s)

    wav_2 = activation(Add()(shortcut), "relu")
    wav_2 = activation(normal(conv1d(wav_2, n_filters, 1, 1)), "relu")

    # softmax损失函数输出结果
    conv_output = activation(normal(conv1d(wav_2, words_size + 1, 1, 1)), "softmax")

    # 模型输出
    wavenet_model = Model(inputs=input_data, outputs=conv_output)

    # 定义CTC模型结构 #
    # 定义CTC模型输入格式 (y_true, dense_output, input_length, label_length)
    y_true = Input(shape=(None,), dtype=np.float32, name="y")
    input_length = Input(shape=(1,), dtype=np.int16, name="X_length")
    label_length = Input(shape=(1,), dtype=np.int16, name="y_length")

    # 定义模型输出格式
    ctc_loss_out = Lambda(ctc_loss, output_shape=(1,), name="ctc")(
        [y_true, conv_output, input_length, label_length]
    )
    # 保存CTC模型结构
    ctc_model = Model(
        inputs=[input_data, y_true, input_length, label_length], outputs=ctc_loss_out
    )

    # 定义模型优化器, 编译模型
    opt_sgd = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True, clipnorm=5)
    ctc_model.compile(
        loss={"ctc": lambda ctc_true, ctc_pred: ctc_pred}, optimizer=opt_sgd
    )

    # 输出模型信息
    ctc_model.summary()

    return wavenet_model, ctc_model


#  3. 分割数据集 / 训练模型

num_mfcc = 20  # mfcc特征维数
test_size = 0.1  # 测试集占比
labels_length = 60  # 标签固定长度
batch_size = 35  # 每批次数据集大小
filter_range = [1, 2, 4, 8, 16]  # 扩大范围
epochs = 280  # 训练次数

# 分段动态学习率
decay_boundaries = [10, 90]  # 学习率迭代回合区间
decay_rates = [0.03, 0.02, 0.01]  # 区间指定学习率
lr_schedule = PiecewiseConstantDecay(boundaries=decay_boundaries, values=decay_rates)

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    train_ds, train_label, test_size=test_size
)

# CTC模型数据generator
train_batch = ctc_batch_generator(
    X_train,
    y_train,
    char2id,
    num_mfcc,
    batch_size=batch_size,
    max_length=labels_length,
)
test_batch = ctc_batch_generator(
    X_test, y_test, char2id, num_mfcc, batch_size=batch_size, max_length=labels_length
)

# 新建模型, 权重保存在wavenet_model中
wavenet_model, ctc_model = model_wavenet(
    len(char2id), num_mfcc, filter_range, learning_rate=lr_schedule
)

# 设置回调函数，在训练验证loss没有继续下降时停止训练
earlystopping = EarlyStopping(
    monitor="val_loss", patience=20, min_delta=1e-5, restore_best_weights=True
)

# 训练模型
history = ctc_model.fit(
    train_batch,
    epochs=epochs,
    callbacks=[earlystopping],
    validation_data=test_batch,
    steps_per_epoch=len(X_train) // batch_size,
    validation_steps=len(X_test) // batch_size,
)

# 保存模型
wavenet_model.save(FILES_PATH + "wavenet.h5")
# 保存训练数据
with open(FILES_PATH + "models/wavenet_history.pkl", "wb") as file:
    pickle.dump(history.history, file)
