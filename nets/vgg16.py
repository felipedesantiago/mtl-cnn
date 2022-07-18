from keras.layers import BatchNormalization, Conv1D, Conv2D, Flatten # ???
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D


def build_vgg16_common_net(input):
    #######################################################################################
    net = Conv2D(name="gender_age_conv_1_1", filters=64, input_shape=(28, 28, 3), kernel_size=(3, 3), padding="same", activation="relu")(input)  # sigmoid
    net = Conv2D(name="gender_age_conv_1_2", filters=64, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = MaxPooling2D(name="gender_age_pool_1", pool_size=(2, 2), strides=(2, 2))(net)
    net = Conv2D(name="gender_age_conv_2_1", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = Conv2D(name="gender_age_conv_2_2", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = MaxPooling2D(name="gender_age_pool_2", pool_size=(2, 2), strides=(2, 2))(net)

    net = Conv2D(name="gender_age_conv_3_1", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = Conv2D(name="gender_age_conv_3_2", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = Conv2D(name="gender_age_conv_3_3", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = MaxPooling2D(name="gender_age_pool_3", pool_size=(2, 2), strides=(2, 2))(net)

    net = Conv2D(name="gender_age_conv_4_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = Conv2D(name="gender_age_conv_4_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = Conv2D(name="gender_age_conv_4_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = MaxPooling2D(name="gender_age_pool_4", pool_size=(2, 2), strides=(2, 2))(net)

    net = Conv2D(name="gender_age_conv_5_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = Conv2D(name="gender_age_conv_5_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = Conv2D(name="gender_age_conv_5_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(net)  # sigmoid
    net = MaxPooling2D(name="gender_age_pool_5", pool_size=(1, 1), strides=(2, 2))(net)  # pool_size SHOULD BE 2,2 but getting error
    #######################################################################################
    return Flatten()(net), Flatten()(net)


def build_vgg16_indep_net(input):
    #######################################################################################
    gender = Conv2D(name="gender_conv_1_1", filters=64, input_shape=(28, 28, 3), kernel_size=(3, 3), padding="same", activation="relu")(input[0])  # sigmoid
    gender = Conv2D(name="gender_conv_1_2", filters=64, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = MaxPooling2D(name="gender_pool_1", pool_size=(2, 2), strides=(2, 2))(gender)
    age = Conv2D(name="age_conv_1_1", filters=64, input_shape=(28, 28, 3), kernel_size=(3, 3), padding="same", activation="relu")(input[1])
    age = Conv2D(name="age_conv_1_2", filters=64, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = MaxPooling2D(name="age_pool_1", pool_size=(2, 2), strides=(2, 2))(age)

    gender = Conv2D(name="gender_conv_2_1", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_2_2", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = MaxPooling2D(name="gender_pool_2", pool_size=(2, 2), strides=(2, 2))(gender)
    age = Conv2D(name="age_conv_2_1", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_2_2", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = MaxPooling2D(name="age_pool_2", pool_size=(2, 2), strides=(2, 2))(age)

    gender = Conv2D(name="gender_conv_3_1", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_3_2", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_3_3", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = MaxPooling2D(name="gender_pool_3", pool_size=(2, 2), strides=(2, 2))(gender)
    age = Conv2D(name="age_conv_3_1", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_3_2", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_3_3", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = MaxPooling2D(name="age_pool_3", pool_size=(2, 2), strides=(2, 2))(age)

    gender = Conv2D(name="gender_conv_4_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_4_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_4_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = MaxPooling2D(name="gender_pool_4", pool_size=(2, 2), strides=(2, 2))(gender)
    age = Conv2D(name="age_conv_4_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_4_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_4_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = MaxPooling2D(name="age_pool_4", pool_size=(2, 2), strides=(2, 2))(age)

    gender = Conv2D(name="gender_conv_5_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_5_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_5_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = MaxPooling2D(name="gender_pool_5", pool_size=(1, 1), strides=(2, 2))(gender)  # pool_size SHOULD BE 2,2 but getting error
    age = Conv2D(name="age_conv_5_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_5_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_5_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = MaxPooling2D(name="age_pool_5", pool_size=(1, 1), strides=(2, 2))(age)
    #######################################################################################
    return Flatten()(gender), Flatten()(age)

def build_vgg16_nddr_net(input):
    def nddr_layer(net_1, net_2):
        net_1 = BatchNormalization()(net_1)
        net_2 = BatchNormalization()(net_2)
        nddr = concatenate([net_1, net_2])
        nddr = Conv1D(filters=1, kernel_size=1, activation="relu")(nddr)
        # nddr = Conv2D(filters=64, kernel_size=(3,3), activation="relu")(nddr)
        return nddr, nddr

    gender = Conv2D(name="gender_conv_1_1", filters=64, input_shape=(28, 28, 3), kernel_size=(3, 3), padding="same", activation="relu")(input)  # sigmoid
    gender = Conv2D(name="gender_conv_1_2", filters=64, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = MaxPooling2D(name="gender_pool_1", pool_size=(2, 2), strides=(2, 2))(gender)
    # if params['inputs'] == 2:
    #     age = Conv2D(name="age_conv_1_1", filters=64, input_shape=(28, 28, 3), kernel_size=(3,3),padding="same",activation="relu")(input2)
    # else:
    #     age = Conv2D(name="age_conv_1_1", filters=64, input_shape=(28, 28, 3), kernel_size=(3,3),padding="same",activation="relu")(input)
    age = Conv2D(name="age_conv_1_1", filters=64, input_shape=(28, 28, 3), kernel_size=(3, 3), padding="same", activation="relu")(input)
    age = Conv2D(name="age_conv_1_2", filters=64, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = MaxPooling2D(name="age_pool_1", pool_size=(2, 2), strides=(2, 2))(age)
    gender, age = nddr_layer(gender, age)

    gender = Conv2D(name="gender_conv_2_1", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_2_2", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = MaxPooling2D(name="gender_pool_2", pool_size=(2, 2), strides=(2, 2))(gender)
    age = Conv2D(name="age_conv_2_1", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_2_2", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = MaxPooling2D(name="age_pool_2", pool_size=(2, 2), strides=(2, 2))(age)
    gender, age = nddr_layer(gender, age)

    gender = Conv2D(name="gender_conv_3_1", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_3_2", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_3_3", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = MaxPooling2D(name="gender_pool_3", pool_size=(2, 2), strides=(2, 2))(gender)
    age = Conv2D(name="age_conv_3_1", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_3_2", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_3_3", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = MaxPooling2D(name="age_pool_3", pool_size=(2, 2), strides=(2, 2))(age)
    gender, age = nddr_layer(gender, age)

    gender = Conv2D(name="gender_conv_4_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_4_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_4_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = MaxPooling2D(name="gender_pool_4", pool_size=(2, 2), strides=(2, 2))(gender)
    age = Conv2D(name="age_conv_4_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_4_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_4_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = MaxPooling2D(name="age_pool_4", pool_size=(2, 2), strides=(2, 2))(age)
    gender, age = nddr_layer(gender, age)

    gender = Conv2D(name="gender_conv_5_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_5_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = Conv2D(name="gender_conv_5_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(gender)  # sigmoid
    gender = MaxPooling2D(name="gender_pool_5", pool_size=(1, 1), strides=(2, 2))(gender)  # pool_size SHOULD BE 2,2 but getting error
    age = Conv2D(name="age_conv_5_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_5_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = Conv2D(name="age_conv_5_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(age)
    age = MaxPooling2D(name="age_pool_5", pool_size=(1, 1), strides=(2, 2))(age)
    gender, age = nddr_layer(gender, age)
    #######################################################################################
    return Flatten()(gender), Flatten()(age)
