"""# Build VGG16 Model"""

from tensorflow.keras.optimizers import Adam as adam_opt
from keras.models import Model
from keras.layers.convolutional import Conv2D # ???
from keras.layers import Conv2D, Input        # ???
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Flatten


def build_vgg16_net():
    trainable_age = NET_TYPE != "VGG_SEPARATED_GENDER"
    trainable_gender = NET_TYPE != "VGG_SEPARATED_AGE"
    input = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, DIMS), name="InputImage")
    #######################################################################################
    net = Conv2D(name="gender_age_conv_1_1", filters=64, input_shape=(28, 28, 3), kernel_size=(3, 3), padding="same",
                 activation="relu")(input)  # sigmoid
    net = Conv2D(name="gender_age_conv_1_2", filters=64, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = MaxPooling2D(name="gender_age_pool_1", pool_size=(2, 2), strides=(2, 2))(net)

    net = Conv2D(name="gender_age_conv_2_1", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = Conv2D(name="gender_age_conv_2_2", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = MaxPooling2D(name="gender_age_pool_2", pool_size=(2, 2), strides=(2, 2))(net)

    net = Conv2D(name="gender_age_conv_3_1", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = Conv2D(name="gender_age_conv_3_2", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = Conv2D(name="gender_age_conv_3_3", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = MaxPooling2D(name="gender_age_pool_3", pool_size=(2, 2), strides=(2, 2))(net)

    net = Conv2D(name="gender_age_conv_4_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = Conv2D(name="gender_age_conv_4_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = Conv2D(name="gender_age_conv_4_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = MaxPooling2D(name="gender_age_pool_4", pool_size=(2, 2), strides=(2, 2))(net)

    net = Conv2D(name="gender_age_conv_5_1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = Conv2D(name="gender_age_conv_5_2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = Conv2D(name="gender_age_conv_5_3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(
        net)  # sigmoid
    net = MaxPooling2D(name="gender_age_pool_5", pool_size=(1, 1), strides=(2, 2))(
        net)  # pool_size SHOULD BE 2,2 but getting error
    #######################################################################################

    age = Flatten()(net)
    gender = Flatten()(net)

    output_gender = Dense(2, activation='softmax', name=NET_TYPE + "_GenderOut", trainable=trainable_gender)(gender)
    output_age = Dense(10, activation='softmax', name=NET_TYPE + "_AgeOut", trainable=trainable_age)(age)

    model = Model(inputs=input, outputs=[output_gender, output_age], name=NET_TYPE)

    opt = adam_opt(learning_rate=LEARNING_RATE)
    # loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction="auto", name="sparse_categorical_crossentropy")
    loss_func = ['categorical_crossentropy', 'categorical_crossentropy']
    print("The COMPILE metric is: " + str(['categorical_crossentropy']))

    # model.compile(loss={NET_TYPE+'_GenderOut':loss_gender, NET_TYPE+'_AgeOut':loss_age}, optimizer=opt, loss_weights=[1, 1], metrics={NET_TYPE+'_GenderOut':'accuracy', NET_TYPE+'_AgeOut':'accuracy'})
    # model.compile(loss={NET_TYPE+'_GenderOut':loss_gender, NET_TYPE+'_AgeOut':loss_age}, optimizer=opt, loss_weights=[gen_weights, age_weights], metrics={NET_TYPE+'_GenderOut':'accuracy', NET_TYPE+'_AgeOut':'accuracy'})
    model.compile(loss=loss_func, optimizer=opt, loss_weights=[gen_weights, age_weights],
                  metrics={NET_TYPE + '_GenderOut': 'accuracy', NET_TYPE + '_AgeOut': 'accuracy'})
    # matriz de confusion
    print(model.summary())
    return model