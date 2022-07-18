from keras import backend as K
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add
from keras.layers import Conv1D, Conv2D, GlobalAveragePooling2D, Reshape, Dense, multiply, Softmax, Flatten
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf

from parameters import *


def build_mn_common_net(input, model_type='large', pooling_type='avg', include_top=True):
    # ** feature extraction layers
    net = __conv2d_block(input, 16, kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS')
    config_list = large_config_list
    for config in config_list:
        net = __bottleneck_block(net, *config)
    # ** final layers
    net = __conv2d_block(net, 960, kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='HS',
                         name='output_map')
    net = GlobalAveragePooling2D()(net) if pooling_type == 'avg' else __global_depthwise_block(net)
    pooled_shape = (1, 1, net.shape[-1])
    net = Reshape(pooled_shape)(net)
    net = Conv2D(1280, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    gender = Conv2D(len(LABELS_GENDER), (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    age = Conv2D(len(LABELS_AGE), (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    return Flatten()(gender), Flatten()(age)

def build_mn_indep_net(input, model_type='large', pooling_type='avg', include_top=True):
    # ** feature extraction layers
    net = __conv2d_block(input[0], 16, kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS')
    net2 = __conv2d_block(input[1], 16, kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS')
    config_list = large_config_list
    for config in config_list:
        net = __bottleneck_block(net, *config)
        net2 = __bottleneck_block(net2, *config)
    # ** final layers
    net = __conv2d_block(net, 960, kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='HS', name='output_map')
    net2 = __conv2d_block(net2, 960, kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='HS', name='output_map2')
    net = GlobalAveragePooling2D()(net) if pooling_type == 'avg' else __global_depthwise_block(net)
    net2 = GlobalAveragePooling2D()(net2) if pooling_type == 'avg' else __global_depthwise_block(net2)
    # ** shape=(None, channel) --> shape(1, 1, channel)
    pooled_shape = (1, 1, net.shape[-1])
    pooled_shape2 = (1, 1, net2.shape[-1])
    net = Reshape(pooled_shape)(net)
    net2 = Reshape(pooled_shape2)(net2)
    net = Conv2D(1280, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    net2 = Conv2D(1280, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net2)
    net2 = Conv2D(len(LABELS_AGE), (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net2)
    net = Conv2D(len(LABELS_GENDER), (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    return Flatten()(net), Flatten()(net2)

def build_mn_nddr_net(input_size, model_type='large', pooling_type='avg', include_top=True):
    def nddr_layer(net_1, net_2):
        net_1 = BatchNormalization()(net_1)
        net_2 = BatchNormalization()(net_2)
        nddr = concatenate([net_1, net_2])
        nddr = Conv1D(filters=1, kernel_size=1, activation="relu")(nddr)
        # nddr = Conv2D(filters=64, kernel_size=(3,3), activation="relu")(nddr)
        return nddr, nddr
    # ** input layer
    input = Input(shape=(input_size, input_size, 3))
    # ** feature extraction layers
    net = __conv2d_block(input, 16, kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS')
    config_list = large_config_list

    for config in config_list:
        net = __bottleneck_block(net, *config)
    # ** final layers
    net = __conv2d_block(net, 960, kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='HS', name='output_map')
    net = GlobalAveragePooling2D()(net) if pooling_type == 'avg' else __global_depthwise_block(net)

    # ** shape=(None, channel) --> shape(1, 1, channel)
    # pooled_shape = (1, 1, net._keras_shape[-1])
    # pooled_shape = (1, 1, net2._keras_shape[-1])
    pooled_shape = (1, 1, net.shape[-1])
    # pooled_shape = (1, 1, net2.shape[-1])

    net = Reshape(pooled_shape)(net)
    # net2 = Reshape(pooled_shape)(net2)
    net = Conv2D(1280, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    # net2 = Conv2D(1280, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net2)

    net2 = Conv2D(len(LABELS_AGE), (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    net = Conv2D(len(LABELS_GENDER), (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)

    net = Flatten()(net)
    net2 = Flatten()(net2)



"""# Build MobileNet Model"""
# Define layers block functions  # https://github.com/godofpdog/MobileNetV3_keras/blob/master/train.py
def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6


# ** update custom Activate functions
get_custom_objects().update({'custom_activation': Activation(Hswish)})

large_config_list = [[16, (3, 3), (1, 1), 16, False, False, False, 'RE', 0],
                     [24, (3, 3), (2, 2), 64, False, False, False, 'RE', 1],
                     [24, (3, 3), (1, 1), 72, False, True, False, 'RE', 2],
                     [40, (5, 5), (2, 2), 72, False, False, True, 'RE', 3],
                     [40, (5, 5), (1, 1), 120, False, True, True, 'RE', 4],
                     [40, (5, 5), (1, 1), 120, False, True, True, 'RE', 5],
                     [80, (3, 3), (2, 2), 240, False, False, False, 'HS', 6],
                     [80, (3, 3), (1, 1), 200, False, True, False, 'HS', 7],
                     [80, (3, 3), (1, 1), 184, False, True, False, 'HS', 8],
                     [80, (3, 3), (1, 1), 184, False, True, False, 'HS', 9],
                     [112, (3, 3), (1, 1), 480, False, False, True, 'HS', 10],
                     [112, (3, 3), (1, 1), 672, False, True, True, 'HS', 11],
                     [160, (5, 5), (1, 1), 672, False, False, True, 'HS', 12],
                     [160, (5, 5), (2, 2), 672, False, True, True, 'HS', 13],
                     [160, (5, 5), (1, 1), 960, False, True, True, 'HS', 14]]

small_config_list = [[16, (3, 3), (2, 2), 16, False, False, True, 'RE', 0],
                     [24, (3, 3), (2, 2), 72, False, False, False, 'RE', 1],
                     [24, (3, 3), (1, 1), 88, False, True, False, 'RE', 2],
                     [40, (5, 5), (1, 1), 96, False, False, True, 'HS', 3],
                     [40, (5, 5), (1, 1), 240, False, True, True, 'HS', 4],
                     [40, (5, 5), (1, 1), 240, False, True, True, 'HS', 5],
                     [48, (5, 5), (1, 1), 120, False, False, True, 'HS', 6],
                     [48, (5, 5), (1, 1), 144, False, True, True, 'HS', 7],
                     [96, (5, 5), (2, 2), 288, False, False, True, 'HS', 8],
                     [96, (5, 5), (1, 1), 576, False, True, True, 'HS', 9],
                     [96, (5, 5), (1, 1), 576, False, True, True, 'HS', 10]]


def __conv2d_block(_inputs, filters, kernel, strides, is_use_bias=False, padding='same', activation='RE', name=None):
    x = Conv2D(filters, kernel, strides=strides, padding=padding, use_bias=is_use_bias)(_inputs)
    x = BatchNormalization()(x)
    x = ReLU(name=name)(x) if activation == 'RE' else Activation(Hswish, name=name)(x)
    return x


def __depthwise_block(_inputs, kernel=(3, 3), strides=(1, 1), activation='RE', is_use_se=True, num_layers=0):
    x = DepthwiseConv2D(kernel_size=kernel, strides=strides, depth_multiplier=1, padding='same')(_inputs)
    x = BatchNormalization()(x)
    if is_use_se:
        x = __se_block(x)
    if activation == 'RE':
        x = ReLU()(x)
    elif activation == 'HS':
        x = Activation(Hswish)(x)
    return x


def __global_depthwise_block(_inputs):
    # assert _inputs._keras_shape[1] == _inputs._keras_shape[2]
    # kernel_size = _inputs._keras_shape[1]
    assert _inputs.shape[1] == _inputs.shape[2]
    kernel_size = _inputs.shape[1]
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(1, 1), depth_multiplier=1, padding='valid')(_inputs)
    return x


def __se_block(_inputs, ratio=4, pooling_type='avg'):
    # filters = _inputs._keras_shape[-1]
    filters = _inputs.shape[-1]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(_inputs) if pooling_type == 'avg' else __global_depthwise_block(
        _inputs)  # else is 'depthwise'
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='hard_sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return multiply([_inputs, se])


def __bottleneck_block(_inputs, out_dim, kernel, strides, expansion_dim, is_use_bias=False, shortcut=True,
                       is_use_se=True, activation='RE', num_layers=0, *args):
    with tf.name_scope('bottleneck_block'):
        # ** to high dim
        bottleneck_dim = expansion_dim

        # ** pointwise conv
        x = __conv2d_block(_inputs, bottleneck_dim, kernel=(1, 1), strides=(1, 1), is_use_bias=is_use_bias,
                           activation=activation)

        # ** depthwise conv
        x = __depthwise_block(x, kernel=kernel, strides=strides, is_use_se=is_use_se, activation=activation,
                              num_layers=num_layers)

        # ** pointwise conv
        x = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)

        if shortcut and strides == (1, 1):
            in_dim = K.int_shape(_inputs)[-1]
            if in_dim != out_dim:
                ins = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(_inputs)
                x = Add()([x, ins])
            else:
                x = Add()([x, _inputs])
    return x


def build_mobilenet_v3(input_size=IMAGE_WIDTH, model_type='large', pooling_type='avg', include_top=True):
    # ** input layer
    input = Input(shape=(input_size, input_size, 3))
    # input2 = Input(shape=(input_size, input_size, 3))

    # ** feature extraction layers
    net = __conv2d_block(input, 16, kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS')
    # if NET_TYPE=="MN_COMMON" or NET_TYPE=="MN_SEPARATED" or NET_TYPE=="MN_SEPARATED_AGE" or NET_TYPE=="MN_SEPARATED_GENDER":
    # net2 = __conv2d_block(input2, 16, kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS')
    # net2 = __conv2d_block(input, 16, kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS')
    config_list = large_config_list

    for config in config_list:
        net = __bottleneck_block(net, *config)
        # net2 = __bottleneck_block(net2, *config)

    # ** final layers
    net = __conv2d_block(net, 960, kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='HS',
                         name='output_map')
    # net2 = __conv2d_block(net2, 960, kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='HS', name='output_map2')
    net = GlobalAveragePooling2D()(net) if pooling_type == 'avg' else __global_depthwise_block(net)
    # net2 = GlobalAveragePooling2D()(net2) if pooling_type == 'avg' else __global_depthwise_block(net2)

    # ** shape=(None, channel) --> shape(1, 1, channel)
    # pooled_shape = (1, 1, net._keras_shape[-1])
    # pooled_shape = (1, 1, net2._keras_shape[-1])
    pooled_shape = (1, 1, net.shape[-1])
    # pooled_shape = (1, 1, net2.shape[-1])

    net = Reshape(pooled_shape)(net)
    # net2 = Reshape(pooled_shape)(net2)
    net = Conv2D(1280, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    # net2 = Conv2D(1280, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net2)

    net2 = Conv2D(len(LABELS_AGE), (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    net = Conv2D(len(LABELS_GENDER), (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)

    net = Flatten()(net)
    net2 = Flatten()(net2)
    net = Softmax(name=NET_TYPE + "_GenderOut")(net)
    net2 = Softmax(name=NET_TYPE + "_AgeOut")(net2)
    losses = ['categorical_crossentropy',
              'categorical_crossentropy']  # ['categorical_crossentropy','categorical_crossentropy']
    model = Model(inputs=input, outputs=[net, net2], name=NET_TYPE + "_Gender/Age")
    model.compile(optimizer=adam_opt(lr=3e-3), loss=losses, metrics=['accuracy'])
    return model
