"""# Train Models"""
# https://towardsdatascience.com/dealing-with-imbalanced-data-in-tensorflow-class-weights-60f876911f99
# https://ichi.pro/es/manejo-de-datos-desequilibrados-en-tensorflow-ponderaciones-de-clase-106620257378201
# https://gist.github.com/angeligareta/83d9024c5e72ac9ebc34c9f0b073c64c

from custom_image_data_generator import CustomImageDataGenerator
from keras.callbacks import ModelCheckpoint
import datetime
from parameters import *

from tensorflow.keras.optimizers import Adam as adam_opt
from keras.models import Model
from keras.layers.convolutional import Conv2D # ???
from keras.layers import Conv2D, Input        # ???
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Flatten
from parameters import *

def train_model(model, train_path, validation_path, net_type, epochs=EPOCHS):  # MAL EL CLASS MODE
    train_data_gen = CustomImageDataGenerator("train", flip_augm=True)  # B
    valid_data_gen = CustomImageDataGenerator("validation", flip_augm=True)  # B
    train_data = train_data_gen.flow(train_path, net_type, BATCH_SIZE)  # B
    valid_data = valid_data_gen.flow(validation_path, net_type, BATCH_SIZE_VAL)  # B
    # validate_batch_VGG_COMMON_GenderOut_accuracy
    # validate_VGG_COMMON_GenderOut_accuracy
    checkpoint = ModelCheckpoint(filepath=MODEL_PATH+net_type+'_model.h5', monitor='val_VGG16_COMMON_AgeOut_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
    print("The TRAIN metric is: " + str(['accuracy']))
    # checkpoint = ModelCheckpoint(filepath=MODEL_PATH+NET_TYPE+'_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)
    start_time = datetime.datetime.now()
    #with EXPERIMENT.train():
    history = model.fit_generator(generator=train_data, validation_data=valid_data, validation_steps=STEPS_VAL, steps_per_epoch=STEPS_EPOCHS, epochs=epochs, callbacks=[checkpoint], verbose=1)
    diff_secs = (datetime.datetime.now() - start_time).total_seconds()
    # RESULTS_DICT[NET_TYPE].append(str(diff_secs / epochs))
    print("Total TRAIN time: " + str(diff_secs) + " - AVG: " + str(diff_secs) + "/" + str(epochs) + " = " + str(diff_secs / epochs))
    return history

def build_model_net(net_type):
    net = None
    input = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, DIMS), name="InputImage")
    if net_type == VGG16_COMMON:
        net = build_vgg16_net(input)
    model = Model(inputs=input, outputs=[net[0], net[1]], name=net_type)

    opt = adam_opt(learning_rate=LEARNING_RATE)

    # loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction="auto", name="sparse_categorical_crossentropy")

    loss_functions = ['categorical_crossentropy', 'categorical_crossentropy']
    model.compile(loss=loss_functions, optimizer=opt, loss_weights=[GEN_WEIGHTS, AGE_WEIGHTS], metrics={net_type + '_GenderOut': 'accuracy', net_type + '_AgeOut': 'accuracy'})

    # loss_functions = {net_type + '_GenderOut': loss_gender, net_type + '_AgeOut': loss_age}
    # model.compile(loss=loss_functions, optimizer=opt, metrics={net_type + '_GenderOut': 'accuracy', net_type + '_AgeOut': 'accuracy'})

    # model.compile(loss={NET_TYPE+'_GenderOut':loss_gender, NET_TYPE+'_AgeOut':loss_age}, optimizer=opt, loss_weights=[1, 1], metrics={NET_TYPE+'_GenderOut':'accuracy', NET_TYPE+'_AgeOut':'accuracy'})
    # model.compile(loss={NET_TYPE+'_GenderOut':loss_gender, NET_TYPE+'_AgeOut':loss_age}, optimizer=opt, loss_weights=[gen_weights, age_weights], metrics={NET_TYPE+'_GenderOut':'accuracy', NET_TYPE+'_AgeOut':'accuracy'})
    # model.compile(loss=[loss_function, loss_function], optimizer=opt, loss_weights=[GEN_WEIGHTS, AGE_WEIGHTS], metrics={net_type + '_GenderOut': 'accuracy', net_type + '_AgeOut': 'accuracy'})
    # matriz de confusion
    print(model.summary())
    return model


def build_vgg16_net(input):
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

    age = Flatten()(net)
    gender = Flatten()(net)

    output_gender = Dense(2, activation='softmax', name=VGG16_COMMON + "_GenderOut", trainable=True)(gender)
    output_age = Dense(10, activation='softmax', name=VGG16_COMMON + "_AgeOut", trainable=True)(age)
    return output_gender, output_age

