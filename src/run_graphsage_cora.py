from tabnanny import verbose
import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model

from graphsage import sample_neighs, GraphSAGE
from utils import preprocess_adj,plot_embeddings, load_data_v1, getsz


def run(a, b, c, d):
    getsz(d)

    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_v1(
        'cora')

    features /= features.sum(axis=1, ).reshape(-1, 1)

    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())

    A = preprocess_adj(A)

    indexs = np.arange(A.shape[0])

    neigh_maxlen = []
    neigh_number = []   #设置采样数
    for i in b:
        i = int(i)
        neigh_number.append(i)

    model_input = [features, np.asarray(indexs, dtype=np.int32)]

    for num in neigh_number:
        sample_neigh, sample_neigh_len = sample_neighs(
            G, indexs, num, self_loop=False)
        sample_neigh = np.asarray(sample_neigh, dtype=np.float32)
        model_input.extend([sample_neigh])
        neigh_maxlen.append(max(sample_neigh_len))

    tf.keras.backend.clear_session()
    model = GraphSAGE(feature_dim=features.shape[1],
                      neighbor_num=neigh_maxlen,
                      n_hidden=16,      #非线性变换参数矩阵的维数
                      n_classes=y_train.shape[1],
                      use_bias=True,
                      activation=tf.nn.relu,
                      aggregator_type=a,
                      dropout_rate=0.5, l2_reg=2.5e-4)
    model.compile(Adam(0.01), 'categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])


    val_data = (model_input, y_val, val_mask)
    mc_callback = ModelCheckpoint('./best_model.h5',
                                  monitor='val_weighted_categorical_crossentropy',
                                  save_best_only=True,
                                  save_weights_only=True)

    #开始训练
    print("start training")
    #validation_data = 验证集, batch_size = 训练一次的样本数, epochs = 总共训练的轮数, verbose = 每一次epoch输出一次日志
    #callbacks = 回调函数(list格式)
    model.fit(model_input, y_train, sample_weight=train_mask, validation_data=val_data,
              batch_size=A.shape[0], epochs=c, shuffle=False, verbose=2,
              callbacks=[mc_callback])

    #载入最佳模型，测试集输出结果
    model.load_weights('./best_model.h5')
    eval_results = model.evaluate(
        model_input, y_test, sample_weight=test_mask, batch_size=A.shape[0])
    #Test loss = 测试误差, Test weighed_loss = 加权损失, accuracy = 准确度
    print('Done.\n'
          'Test loss: {}\n'
          'Test weighted_loss: {}\n'
          'Test accuracy: {}'.format(*eval_results))
    return eval_results


def classify():

    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_v1('cora')

    features /= features.sum(axis=1, ).reshape(-1, 1)

    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())

    A = preprocess_adj(A)

    indexs = np.arange(A.shape[0])
    neigh_number = [10, 25]  # 设置采样数
    neigh_maxlen = []

    model_input = [features, np.asarray(indexs, dtype=np.int32)]

    for num in neigh_number:
        sample_neigh, sample_neigh_len = sample_neighs(
            G, indexs, num, self_loop=False)
        sample_neigh = np.asarray(sample_neigh, dtype=np.float32)
        model_input.extend([sample_neigh])
        neigh_maxlen.append(max(sample_neigh_len))

    tf.keras.backend.clear_session()
    model = GraphSAGE(feature_dim=features.shape[1],
                      neighbor_num=neigh_maxlen,
                      n_hidden=16,  # 非线性变换参数矩阵的维数
                      n_classes=y_train.shape[1],
                      use_bias=True,
                      activation=tf.nn.relu,
                      aggregator_type='mean',
                      dropout_rate=0.5, l2_reg=2.5e-4)
    model.compile(Adam(0.01), 'categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])

    #可视化结果展示
    #tf.keras.backend.clear_session()
    model.load_weights('./best_model.h5')
    gcn_embedding = model.layers[-1]
    embedding_model = Model(model.input, outputs=Lambda(lambda x: gcn_embedding.output)(model.input))
    embedding_weights = embedding_model.predict(model_input, batch_size=A.shape[0])
    y  = np.genfromtxt("{}{}.content".format('../data/cora/', 'cora'), dtype=np.dtype(str))[:, -1]
    plot_embeddings(embedding_weights, np.arange(A.shape[0]), y)

