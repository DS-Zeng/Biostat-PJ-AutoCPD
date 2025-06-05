import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import layers, losses, metrics, models


def general_simple_nn(n, l, m, num_classes, model_name="simple_nn"):
    """
    To construct a simple neural network.

    Parameters
    ----------
    n : scalar
        the input size
    l : scalar
        the number of hidden layers
    m : scalar or 1D array
        the width vector of hidden layers, if it is a scalar, then the hidden layers of simple neural network have the same nodes.
    num_classes : scalar
        the nodes of output layers, i.e., the number of classes
    model_name : str, optional
        the model name, by default "simple_nn"

    Returns
    -------
    model
        the simple neural network
    """
    input_layer = layers.Input(shape=(n,), name="Input")
    if isinstance(m, int):
        m_vec = np.repeat(m, l)
    elif len(m) == l:
        m_vec = m
    else:
        warnings.warn(
            "The length of width vector must be equal to the number of hidden layers.",
            DeprecationWarning,
        )

    x = layers.Dense(m_vec[0], activation="relu", kernel_regularizer="l2")(input_layer)
    if l >= 2:
        for k in range(l - 1):
            x = layers.Dense(m_vec[k + 1], activation="relu", kernel_regularizer="l2")(
                x
            )

    output_layer = layers.Dense(num_classes)(x)
    model = models.Model(input_layer, output_layer, name=model_name)
    return model


def general_transformer_nn(
    n, 
    d_model=64, 
    num_heads=4, 
    ff_dim=128, 
    num_layers=1,
    num_classes=2, 
    dropout_rate=0.1,
    model_name="transformer_nn"
):
    """
    To construct a transformer neural network for time series classification.

    Parameters
    ----------
    n : int
        the input sequence length (time series length)
    d_model : int, optional
        the dimension of the model, by default 64
    num_heads : int, optional
        the number of attention heads, by default 4
    ff_dim : int, optional
        the dimension of the feed-forward network, by default 128
    num_layers : int, optional
        the number of transformer layers, by default 1
    num_classes : int
        the number of output classes
    dropout_rate : float, optional
        the dropout rate, by default 0.1
    model_name : str, optional
        the model name, by default "transformer_nn"

    Returns
    -------
    model
        the transformer neural network
    """
    inputs = tf.keras.Input(shape=(n,), name="Input")
    
    # Reshape and embed input
    x = layers.Reshape((n, 1))(inputs)
    x = layers.Dense(d_model)(x)
    
    # Add positional encoding
    pos_emb = layers.Embedding(input_dim=n, output_dim=d_model)
    positions = tf.range(start=0, limit=n, delta=1)
    x += pos_emb(positions)
    
    # Transformer layers
    for _ in range(num_layers):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model,
            dropout=dropout_rate
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward network
        ffn = layers.Dense(ff_dim, activation="relu")(x)
        ffn = layers.Dropout(dropout_rate)(ffn)
        ffn = layers.Dense(d_model)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization()(x)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes)(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name=model_name)


def general_transformer_nn_v2(
    time_steps, 
    input_dim,
    d_model=128, 
    num_heads=4, 
    ff_dim=256, 
    num_layers=1,
    num_classes=2, 
    dropout_rate=0.1,
    model_name="transformer_nn_v2"
):
    """
    构建改进的Transformer神经网络，直接接收(batch, time_steps, features)格式的输入
    更符合Transformer原始设计理念：时间步作为序列长度，特征维度作为嵌入维度

    Parameters
    ----------
    time_steps : int
        时间步数（序列长度）
    input_dim : int
        每个时间步的特征维度
    d_model : int, optional
        Transformer模型维度, by default 128
    num_heads : int, optional
        注意力头数, by default 4
    ff_dim : int, optional
        前馈网络维度, by default 256
    num_layers : int, optional
        Transformer层数, by default 1
    num_classes : int
        分类数量
    dropout_rate : float, optional
        Dropout率, by default 0.1
    model_name : str, optional
        模型名称, by default "transformer_nn_v2"

    Returns
    -------
    model
        改进的Transformer神经网络模型
    """
    # 输入层：(batch, time_steps, input_dim)
    inputs = tf.keras.Input(shape=(time_steps, input_dim), name="Input")
    
    # 如果输入特征维度与模型维度不同，需要线性投影
    if input_dim != d_model:
        x = layers.Dense(d_model, name="input_projection")(inputs)
    else:
        x = inputs
    
    # 添加位置编码
    pos_emb = layers.Embedding(input_dim=time_steps, output_dim=d_model, name="positional_embedding")
    positions = tf.range(start=0, limit=time_steps, delta=1)
    x += pos_emb(positions)
    
    # Transformer编码器层
    for i in range(num_layers):
        # 多头自注意力
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads,  # 每个头的维度
            dropout=dropout_rate,
            name=f"multihead_attention_{i}"
        )(x, x)
        
        # 残差连接和层归一化
        x = layers.Add(name=f"add_attention_{i}")([x, attn_output])
        x = layers.LayerNormalization(name=f"layer_norm_attention_{i}")(x)
        
        # 前馈网络
        ffn = layers.Dense(ff_dim, activation="relu", name=f"ffn_dense1_{i}")(x)
        ffn = layers.Dropout(dropout_rate, name=f"ffn_dropout_{i}")(ffn)
        ffn = layers.Dense(d_model, name=f"ffn_dense2_{i}")(ffn)
        
        # 残差连接和层归一化
        x = layers.Add(name=f"add_ffn_{i}")([x, ffn])
        x = layers.LayerNormalization(name=f"layer_norm_ffn_{i}")(x)
    
    # 全局平均池化，将序列维度压缩为单一向量
    x = layers.GlobalAveragePooling1D(name="global_avg_pooling")(x)
    
    # 分类前的Dropout
    x = layers.Dropout(dropout_rate, name="final_dropout")(x)
    
    # 输出层
    outputs = layers.Dense(num_classes, name="classification_head")(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name=model_name)


# mymodel = simple_nn(n=100, l=1, m=10, num_classes=2)
# mymodel = simple_nn(n=100, l=3, m=10, num_classes=2)
# mymodel = simple_nn(n=100, l=3, m=[20, 20, 5], num_classes=2)

# build the model, train and save it to disk


def get_optimizer(learning_rate):
    """
    To get the optimizer given the learning rate.

    Parameters
    ----------
    learning_rate : float
        the learning rate for inverse time decay schedule.

    Returns
    -------
    optimizer
        the Adam
    """
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        learning_rate, decay_steps=10000, decay_rate=1, staircase=False
    )
    return tf.keras.optimizers.Adam(lr_schedule)


def get_callbacks(name, log_dir, epochdots):
    """
    Get callbacks. This function returns the result of epochs during training, if it satisfies some conditions then the training can stop early. At meanwhile, this function also save the results of training in TensorBoard and csv files.

    Parameters
    ----------
    name : str
        the model name
    log_dir : str
        the path of log files
    epochdots : object
        the EpochDots object from tensorflow_docs

    Returns
    -------
    list
        the list of callbacks
    """
    name1 = name + "/log.csv"
    return [
        epochdots,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_crossentropy", patience=800, min_delta=1e-3
        ),
        tf.keras.callbacks.TensorBoard(Path(log_dir, name)),
        tf.keras.callbacks.CSVLogger(Path(log_dir, name1)),
    ]


def compile_and_fit(
    model,
    x_train,
    y_train,
    batch_size,
    lr,
    name,
    log_dir,
    epochdots,
    optimizer=None,
    validation_split=0.2,
    max_epochs=10000,
):
    """
    To compile and fit the model

    Parameters
    ----------
    model : Models object
        the simple neural network
    x_train : tf.Tensor
        the tensor of training data
    y_train : tf.Tensor
        the tensor of training data, label
    batch_size : int
        the batch size
    lr : float
        the learning rate
    name : str
        the model name
    log_dir : str
        the path of log files
    epochdots : object
        the EpochDots object from tensorflow_docs
    optimizer : optimizer object or str, optional
        the optimizer, by default None
    max_epochs : int, optional
        the maximum number of epochs, by default 10000

    Returns
    -------
    model.fit object
        a fitted model object
    """
    if optimizer is None:
        optimizer = get_optimizer(lr)
    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            metrics.SparseCategoricalCrossentropy(
                from_logits=True, name="sparse_categorical_crossentropy"
            ),
            "accuracy",
        ],
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=get_callbacks(name, log_dir, epochdots),
        verbose=2,
    )
    return history


def resblock(x, kernel_size, filters, strides=1):
    """
    This function constructs a resblock.

    Parameters
    ----------
    x : tensor
        the input data
    kernel_size : int
        the kernel size
    filters : int
        the filter size
    strides : int, optional
        the stride, by default 1

    Returns
    -------
    layer
        the hidden layer
    """
    x1 = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv2D(filters, kernel_size, padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    if strides != 1:
        x = layers.Conv2D(filters, 1, strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)

    x1 = layers.Add()([x, x1])
    x1 = layers.ReLU()(x1)
    return x1


def deep_nn(
    n,
    n_trans,
    kernel_size,
    n_filter,
    dropout_rate,
    n_classes,
    m,
    l,
    model_name="deep_nn",
):
    """
    This function is used to construct the deep neural network with 21 residual blocks.

    Parameters
    ----------
    n : int
        the length of time series
    n_trans : int
        the number of transformations
    kernel_size : int
        the kernel size
    n_filter : int
        the filter size
    dropout_rate : float
        the dropout rate
    n_classes : int
        the number of classes
    m : array
        the width vector
    l : int
        the number of dense layers

    model_name : str, optional
        the model name, by default "deep_nn"

    Returns
    -------
    model
        the model of deep neural network
    """
    # Note: the following network will cost several hours to train the residual neural network in GPU server.
    input_layer = layers.Input(shape=(n_trans, n), name="Input")
    x = layers.Reshape((n_trans, n, 1))(input_layer)
    x = layers.Conv2D(n_filter, 2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)

    x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)

    x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)

    x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)

    x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)

    x = layers.GlobalAveragePooling2D()(x)
    for i in range(l - 1):
        x = layers.Dense(m[i], activation="relu", kernel_regularizer="l2")(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(m[l - 1], activation="relu", kernel_regularizer="l2")(x)
    output_layer = layers.Dense(n_classes)(x)
    model = models.Model(input_layer, output_layer, name=model_name)
    return model


def general_deep_nn(
    n,
    n_trans,
    kernel_size,
    n_filter,
    dropout_rate,
    n_classes,
    n_resblock,
    m,
    l,
    model_name="deep_nn",
):
    """
    This function is used to construct the deep neural network with 21 residual blocks.

    Parameters
    ----------
    n : int
        the length of time series
    n_trans : int
        the number of transformations
    kernel_size : int
        the kernel size
    n_filter : int
        the filter size
    dropout_rate : float
        the dropout rate
    n_classes : int
        the number of classes
    n_resnet : int
        the number of residual blocks
    m : array
        the width vector
    l : int
        the number of dense layers
    model_name : str, optional
        the model name, by default "deep_nn"

    Returns
    -------
    model
        the model of deep neural network
    """
    # Note: the following network will cost several hours to train the residual neural network in GPU server.
    input_layer = layers.Input(shape=(n_trans, n), name="Input")
    x = layers.Reshape((n_trans, n, 1))(input_layer)
    x = layers.Conv2D(n_filter, 2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    j1 = n_resblock % 4
    for _ in range(j1):
        x = resblock(x, kernel_size, filters=n_filter)
    j2 = n_resblock // 4
    if j2 > 0:
        for _ in range(j2):
            x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
            x = resblock(x, kernel_size, filters=n_filter)
            x = resblock(x, kernel_size, filters=n_filter)
            x = resblock(x, kernel_size, filters=n_filter)

    x = layers.GlobalAveragePooling2D()(x)
    for i in range(l - 1):
        x = layers.Dense(m[i], activation="relu", kernel_regularizer="l2")(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(m[l - 1], activation="relu", kernel_regularizer="l2")(x)
    output_layer = layers.Dense(n_classes)(x)
    model = models.Model(input_layer, output_layer, name=model_name)
    return model


def general_deep_nn_4d(
    channels,
    height,
    width,
    kernel_size,
    n_filter,
    dropout_rate,
    n_classes,
    n_resblock,
    m,
    l,
    model_name="deep_nn_4d",
):
    """
    This function is used to construct the deep neural network for 4D input (channels, height, width).

    Parameters
    ----------
    channels : int
        the number of input channels (dimensions)
    height : int
        the height of input data
    width : int
        the width of input data
    kernel_size : tuple
        the kernel size
    n_filter : int
        the filter size
    dropout_rate : float
        the dropout rate
    n_classes : int
        the number of classes
    n_resblock : int
        the number of residual blocks
    m : array
        the width vector
    l : int
        the number of dense layers
    model_name : str, optional
        the model name, by default "deep_nn_4d"

    Returns
    -------
    model
        the model of deep neural network for 4D input
    """
    # Input layer for 4D data: (batch, channels, height, width)
    input_layer = layers.Input(shape=(channels, height, width), name="Input")
    
    # First conv layer
    x = layers.Conv2D(n_filter, kernel_size, padding="same")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual blocks
    j1 = n_resblock % 4
    for _ in range(j1):
        x = resblock(x, kernel_size, filters=n_filter)
    j2 = n_resblock // 4
    if j2 > 0:
        for _ in range(j2):
            x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
            x = resblock(x, kernel_size, filters=n_filter)
            x = resblock(x, kernel_size, filters=n_filter)
            x = resblock(x, kernel_size, filters=n_filter)

    # Global pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    for i in range(l - 1):
        x = layers.Dense(m[i], activation="relu", kernel_regularizer="l2")(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(m[l - 1], activation="relu", kernel_regularizer="l2")(x)
    output_layer = layers.Dense(n_classes)(x)
    model = models.Model(input_layer, output_layer, name=model_name)
    return model
