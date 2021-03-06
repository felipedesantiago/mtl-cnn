# -*- coding: utf-8 -*-
"""VGG16 Common.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mg240dE2EJ9Ujwn5VEIqPuf_-L_aqy6G

# Installs & Imports
"""

# datastes: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
import comet_ml
# from comet_ml import Experiment
import cv2
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.optimizers import Adam as adam_opt
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from keras.layers.pooling import MaxPooling2D, MaxPooling3D
from keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Softmax, Flatten
from tensorflow.keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import losses
from keras import backend as K
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import image
from PIL import Image as pil_image
from random import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

from IPython.display import Image, display
from tensorflow.keras.datasets.imdb import load_data
from tensorflow.keras.datasets.reuters import load_data
from google.colab import drive
import datetime
import csv

import traceback

"""# Models & Parameters"""
# loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction="auto", name="sparse_categorical_crossentropy")
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

loss_func = CategoricalCrossentropy(name="categorical_crossentropy")
metr_func = tf.keras.metrics.CategoricalAccuracy()  # name='sparse_categorical_accuracy', dtype=None)
losses_1 = loss_func
losses_2 = [loss_func, loss_func]
# params['net_name']
tf.keras.metrics.Accuracy()
vgg_paralel_acc = CategoricalAccuracy("VGG_PARALEL_GenderOut_Accuracy")

"""# Global Variables"""

# PARAMETERS

LEARNING_RATE = 0.0001

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

TARGET_WIDTH = 256
TARGET_HEIGHT = 256
DIMS = 3

TRAIN_VAL_RATIO = 5
TRAIN_VAL_CERO = 0

BATCH_SIZE = 32
BATCH_SIZE_VAL = 32
# Make sure that your iterator can generate at least `steps_per_epoch * epochs` batches
EPOCHS = 40
STEPS_EPOCHS = 1182  # training images #trainimages / batch_size * 2 (2 because of DA flip)
STEPS_VAL = 294  # valid images #trainimages / batch_size * 2 (2 because of DA flip)

DATA_PATH = "gdrive/MyDrive/ColabNotebooks/images/datasets/UTKFace/"
MODEL_PATH = "gdrive/MyDrive/ColabNotebooks/models/"
DO_TRAIN = True
PLOT_MODELS = True
NET_TYPE = None

LABELS_GENDER = ['male', 'female']
LABELS_GENDER_SMALL = ['m', 'f']
LABELS_AGE = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']

TOTALS = {"genders": {0: 0, 1: 0}, "ages": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}}

MAX_PREDICTION_IMAGES = 75  # 75 #MAX: 78

MODEL_COMMON = None
MODEL_NDDR = None
MODEL_PARALEL = None
MODEL_SEPARATED = None
MODEL_SEPARATED_GENDER = None
MODEL_SEPARATED_AGE = None

RESULTS_FILENAME = None
RESULTS_DICT = None

EXPERIMENT = None

"""# Auxiliar Functions""
def print_batch(x1, y1, y2, with_prints=False):
    if with_prints:
        print("X1 shape: " + str(x1[0].shape))
        print("Gender generator - " + str(y1))  # matrix_as_shape_list(y1))
        print("Age generator - " + str(y2))  # matrix_as_shape_list(y2))
        print(to_categorical(y1, num_classes=2).shape)
        print(to_categorical(y2, num_classes=10).shape)


def matrix_as_shape_list(the_matrix):
    res = [item for sublist in the_matrix for item in sublist]  # flatten array
    return " shape: " + str(the_matrix.shape) + " - " + " ".join(map(str, res))


def plot_history(history):
    # {'loss': [2.7617995738983154], 'GenderOut_loss': [0.6187705993652344], 'AgeOut_loss': [2.143028974533081], 'GenderOut_accuracy': [0.7515624761581421], 'AgeOut_accuracy': [0.2515625059604645]}
    # Agregar validaci??n a las gr??ficas
    # val_loss - val_GenderOut_loss - val_AgeOut_loss - val_GenderOut_accuracy - val_AgeOut_accuracy

    # print(d["thekey"]) if "the key" in d else None
    print("The history: " + str(history.history))
    plt.style.use('seaborn-whitegrid')
    plt.axes()
    plt.plot(history.history['val_' + NET_TYPE + '_GenderOut_accuracy'], 'r',
             label="GenderVAL") if 'val_' + NET_TYPE + '_GenderOut_accuracy' in history.history else None
    # plt.plot(history.history[NET_TYPE+'_GenderOut_accuracy'],'m', label="GenderTRAIN") if NET_TYPE+'_GenderOut_accuracy' in history.history else None
    plt.plot(history.history['val_' + NET_TYPE + '_AgeOut_accuracy'], 'b',
             label="AgeVAL") if 'val_' + NET_TYPE + '_AgeOut_accuracy' in history.history else None
    # plt.plot(history.history[NET_TYPE+'_AgeOut_accuracy'],'c', label="AgeTRAIN") if NET_TYPE+'_AgeOut_accuracy' in history.history else None
    plt.plot(history.history['accuracy'], 'm', label="TRAIN") if 'accuracy' in history.history else None
    plt.plot(history.history['val_accuracy'], 'r', label="VAL") if 'val_accuracy' in history.history else None
    plt.title('Accuracy ' + NET_TYPE)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    # summarize history for loss
    plt.style.use('seaborn-whitegrid')
    plt.axes()
    plt.plot(history.history['loss'], 'g', label="Combined") if 'loss' in history.history else None
    plt.plot(history.history['val_' + NET_TYPE + '_GenderOut_loss'], 'r',
             label="GenderVAL") if 'val_' + NET_TYPE + '_GenderOut_loss' in history.history else None
    # plt.plot(history.history[NET_TYPE+'_GenderOut_loss'],'m', label="GenderTRAIN") if NET_TYPE+'_GenderOut_loss' in history.history else None
    plt.plot(history.history['val_' + NET_TYPE + '_AgeOut_loss'], 'b',
             label="AgeVAL") if 'val_' + NET_TYPE + '_AgeOut_loss' in history.history else None
    # plt.plot(history.history[NET_TYPE+'_AgeOut_loss'],'c', label="AgeTRAIN") if NET_TYPE+'_AgeOut_loss' in history.history else None
    plt.plot(history.history['loss'], 'm', label="TRAIN") if 'loss' in history.history else None
    plt.plot(history.history['val_loss'], 'r', label="VAL") if 'val_loss' in history.history else None
    plt.title('Loss ' + NET_TYPE)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def predict_images(model, max_imgs=MAX_PREDICTION_IMAGES, with_prints=False):
    preview_size = 50
    print("Predicting...")
    print(TOTALS)
    predicted = 0
    pred_accuracy = {"gender": [0, 0], "age": [0, 0]}
    gen_idx = 0
    age_idx = 1
    if "SEPARATED_AGE" in NET_TYPE:
        age_idx = 0
    PREDICTION_PATHS = "gdrive/MyDrive/ColabNotebooks/images/datasets/predict/"
    start_time = datetime.datetime.now()
    y_true = [[], []]
    y_pred = [[], []]
    for img_path in os.listdir(PREDICTION_PATHS):
        # TRY to do a model.predict() with a set or list of images, instead of one by one?
        if with_prints:
            print("\nImage: " + str(img_path))
            display(Image(os.path.join(PREDICTION_PATHS, img_path), width=preview_size, height=preview_size))

        img = cv2.imread(os.path.join(PREDICTION_PATHS, img_path))
        # img = img[..., ::-1]
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = np.expand_dims(img, axis=0)
        # print("predicting the net: "+NET_TYPE)
        prediction = model.predict(img)

        preds = "Gender preds: "
        for it in range(0, 2):
            preds = preds + " " + str(prediction[gen_idx].item(it))
        print("GEN ERROR: " + str(
            not int(img_path.startswith(LABELS_GENDER_SMALL[np.argmax(prediction[gen_idx])]))) + " - Gender: " +
              LABELS_GENDER[np.argmax(prediction[gen_idx])] + " - " + str(preds)) if with_prints else None
        pred_accuracy["gender"][0] += int(not img_path.startswith(LABELS_GENDER_SMALL[np.argmax(prediction[gen_idx])]))
        pred_accuracy["gender"][1] += 1
        y_true[0].append(LABELS_GENDER_SMALL.index(img_path.split("_")[0]))
        y_pred[0].append(np.argmax(prediction[gen_idx]))

        preds = "Age preds: "
        age_posta = int(img_path.split("_")[1]) + 5
        for it in range(0, 10):
            preds = preds + " " + str(prediction[age_idx].item(it))
        print("AGE ERROR: " + str(abs(age_posta - int(LABELS_AGE[np.argmax(prediction[age_idx])]))) + " - Age: " +
              LABELS_AGE[np.argmax(prediction[age_idx])] + " - " + str(preds)) if with_prints else None
        pred_accuracy["age"][0] += abs(age_posta - int(LABELS_AGE[np.argmax(prediction[age_idx])])) + 5
        pred_accuracy["age"][1] += 1
        y_true[1].append(age_posta)
        y_pred[1].append(int(LABELS_AGE[np.argmax(prediction[age_idx])]) + 5)
        if predicted == max_imgs:
            break
        else:
            predicted += 1
    diff_secs = (datetime.datetime.now() - start_time).total_seconds()
    print(str(pred_accuracy))
    print("Gender error average: " + str(
        pred_accuracy["gender"][0] / pred_accuracy["gender"][1])) if "SEPARATED_AGE" not in NET_TYPE else print(
        "Gender error average: -")
    print("Age error average: " + str(
        pred_accuracy["age"][0] / pred_accuracy["age"][1])) if "SEPARATED_GENDER" not in NET_TYPE else print(
        "Age error average: -")
    RESULTS_DICT[NET_TYPE].append(str(diff_secs / max_imgs))
    RESULTS_DICT[NET_TYPE].append(
        str(pred_accuracy["gender"][0] / pred_accuracy["gender"][1])) if "SEPARATED_AGE" not in NET_TYPE else \
    RESULTS_DICT[NET_TYPE].append("")
    RESULTS_DICT[NET_TYPE].append(
        str(pred_accuracy["age"][0] / pred_accuracy["age"][1])) if "SEPARATED_GENDER" not in NET_TYPE else RESULTS_DICT[
        NET_TYPE].append("")
    print("Total PREDICTION time: " + str(diff_secs) + " - AVG: " + str(diff_secs) + "/" + str(max_imgs) + " = " + str(
        diff_secs / max_imgs))
    print("Confusion matriz: ")
    print("Gender true: " + str(y_true))
    print("Gender pred: " + str(y_pred))
    # mapped_true = [int(round(a/10)) for a in y_true[1]] # map(lambda a : a/float(10), y_true[1])
    mapped_true = [int(floor(a / 10)) for a in y_true[1]]
    mapped_pred = [int(floor(a / 10)) for a in y_pred[1]]
    # mapped_pred = [int(round(a/10)) for a in y_pred[1]] # map(lambda a : a/float(10), y_pred[1])
    print("Age true " + str(mapped_true))
    print("Age pred " + str(mapped_pred))
    experiment = comet_ml.Experiment(
        api_key="6rubQB46mQ6XTVLSl26nH6zpV",
        project_name="Predictions",
    )
    experiment.log_confusion_matrix(y_true[0], y_pred[0], title=NET_TYPE + " Gender",
                                    file_name=NET_TYPE + "_Gender.json", labels=["Male", "Female"])
    experiment.log_confusion_matrix(mapped_true, mapped_pred, title=NET_TYPE + " Age",
                                    file_name=NET_TYPE + "_Age.json")  # , labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", ">90"])
    experiment.end()
    # label = decode_predictions(prediction)


def create_results_filename():
    now_date = datetime.datetime.now()
    filename = MODEL_PATH + now_date.strftime("%B") + "_" + now_date.strftime("%d") + "_" + now_date.strftime(
        "%H") + "_" + now_date.strftime("%M") + "_" + "results.csv"
    return filename


def create_header():
    return ['Headers', 'Epochs', 'Train steps', 'Val steps', 'Batch Size', 'Traintime', 'Model Param count',
            'Prediction time', 'Gender Custom Error', 'Age Custom Error', 'Gender Accuracy', 'Age Accuracy',
            'Gender Loss', 'Age Loss', 'Combined Loss', 'Best at Epoch', 'Time per Epoch', 'Model Size']


def write_to_file(filename):
    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile)
        for key in RESULTS_DICT.keys():
            writer.writerow(RESULTS_DICT[key])
        csvfile.close()
    print("Updated results in: " + str(filename))


"""# Load & Save Model"""


def save_model(my_model):
    print("SAVE DATA:")
    print(str(MODEL_PATH))
    print(str(NET_TYPE))
    my_model.save(filepath=MODEL_PATH + NET_TYPE + "2_model", overwrite=True)
    print("Saved model: " + MODEL_PATH + NET_TYPE + "2_model")


def load_model_from_disk():
    print("Loading model: " + str(MODEL_PATH) + str(NET_TYPE) + "2_model")
    to_ret = load_model(filepath=MODEL_PATH + NET_TYPE + "2_model")
    print("Loaded model: " + str(MODEL_PATH) + str(NET_TYPE) + "2_model")
    return to_ret


"""# Train Models"""

# https://towardsdatascience.com/dealing-with-imbalanced-data-in-tensorflow-class-weights-60f876911f99
# https://ichi.pro/es/manejo-de-datos-desequilibrados-en-tensorflow-ponderaciones-de-clase-106620257378201
# https://gist.github.com/angeligareta/83d9024c5e72ac9ebc34c9f0b073c64c

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer


def generate_class_weights(class_series, multi_class=True, one_hot_encoded=True):
    """
    Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.
    Some examples of different formats of class_series and their outputs are:
      - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
      {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
      - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
      {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
      - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
      {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
      - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
      {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}
    The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
    of appareance of the label when the dataset was processed.
    In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).
    Author: Angel Igareta (angel@igareta.com)
    """
    if multi_class:
        # If class is one hot encoded, transform to categorical labels to use compute_class_weight
        if one_hot_encoded:
            class_series = np.argmax(class_series, axis=1)

        # Compute class weights with sklearn method
        class_labels = np.unique(class_series)
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
        return dict(zip(class_labels, class_weights))
    else:
        # It is neccessary that the multi-label values are one-hot encoded
        mlb = None
        if not one_hot_encoded:
            mlb = MultiLabelBinarizer()
            class_series = mlb.fit_transform(class_series)

        n_samples = len(class_series)
        n_classes = len(class_series[0])

        # Count each class frequency
        class_count = [0] * n_classes
        for classes in class_series:
            for index in range(n_classes):
                if classes[index] != 0:
                    class_count[index] += 1

        # Compute class weights using balanced method
        class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
        class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
        return dict(zip(class_labels, class_weights))


import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
class_weights = generate_class_weights(train_labels, one_hot_encoded=False)  # Previously imported
n_classes = len(class_weights)
print(str(class_weights))
print(str(n_classes))


# Method by Morten Gr??ftehauge. # Source: https://github.com/keras-team/keras/issues/11735#issuecomment-641775516

def weighted_categorical_crossentropy(class_weight):
    """Returns a loss function for a specific class weight tensor
    Params:
        class_weight: 1-D constant tensor of class weights
    Returns:
        A loss function where each loss is scaled according to the observed class"""

    def loss(y_obs, y_pred):
        y_obs = tf.dtypes.cast(y_obs, tf.int32)
        hothot = tf.one_hot(tf.reshape(y_obs, [-1]), depth=len(class_weight))
        weight = tf.math.multiply(class_weight, hothot)
        weight = tf.reduce_sum(weight, axis=-1)
        # losses = tf.keras.losses.categorical_crossentropy(sample_weight=weight)
        losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y_obs, logits=y_pred, weights=weight)
        # losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(weights=weight)
        return losses

    return loss


# Compile loss function to be used for both model outputs
loss = weighted_categorical_crossentropy(list(class_weights.values()))


# Build model with Keras Functional API
def build_model():
    input_layer = tf.keras.layers.Input(shape=(28, 28))
    dense_layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    dense_layer = tf.keras.layers.Flatten()(dense_layer)
    output_layer_1 = tf.keras.layers.Dense(n_classes, name="output_1")(dense_layer)
    output_layer_2 = tf.keras.layers.Dense(n_classes, name="output_2")(dense_layer)
    model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer_1, output_layer_2])
    return model


# Build and compile model specifying custom loss and metrics for both outputs
model = build_model()
model.compile(optimizer="adam", loss={"output_1": loss, "output_2": loss},
              metrics={"output_1": "accuracy", "output_2": "accuracy"})
# Just for demo purposes, add the same label as second output
train_labels = (train_labels, train_labels)
test_labels = (test_labels, test_labels)
# Train and evaluate the model
model.fit(train_images, train_labels, epochs=2)
result = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy_output 1:", result[3])
print("\nTest accuracy output 2:", result[4])

age_weights = {0: 1, 1: 1.8446, 2: 0.5949, 3: 0.3593, 4: 0.61249, 5: 1.2032, 6: 1.2199, 7: 3.39511, 8: 3.3142,
               9: 8.4695}
gen_weights = {0: 0.955, 1: 1.049}

loss_gender = weighted_categorical_crossentropy(list(gen_weights.values()))
loss_age = weighted_categorical_crossentropy(list(age_weights.values()))
print("FIN")


def train_model(model, train_path, validation_path, epochs=EPOCHS):  # MAL EL CLASS MODE
    train_data_gen = CustomImageDataGenerator("train", flip_augm=True)  # B
    valid_data_gen = CustomImageDataGenerator("validation", flip_augm=True)  # B
    train_data = train_data_gen.flow(train_path, [], BATCH_SIZE)  # B
    valid_data = valid_data_gen.flow(validation_path, [], BATCH_SIZE_VAL)  # B
    # validate_batch_VGG_COMMON_GenderOut_accuracy
    # validate_VGG_COMMON_GenderOut_accuracy
    checkpoint = ModelCheckpoint(filepath=MODEL_PATH + NET_TYPE + '_model.h5',
                                 monitor='val_VGG16_COMMON_AgeOut_accuracy', verbose=1, save_best_only=True, mode='max',
                                 period=1)
    print("The TRAIN metric is: " + str(['accuracy']))
    # checkpoint = ModelCheckpoint(filepath=MODEL_PATH+NET_TYPE+'_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)
    start_time = datetime.datetime.now()
    with EXPERIMENT.train():
        history = model.fit_generator(generator=train_data, validation_data=valid_data, validation_steps=STEPS_VAL,
                                      steps_per_epoch=STEPS_EPOCHS, epochs=epochs, callbacks=[checkpoint], verbose=1)
    diff_secs = (datetime.datetime.now() - start_time).total_seconds()
    RESULTS_DICT[NET_TYPE].append(str(diff_secs / epochs))
    print("Total TRAIN time: " + str(diff_secs) + " - AVG: " + str(diff_secs) + "/" + str(epochs) + " = " + str(
        diff_secs / epochs))
    return history


"""# Build VGG16 Model"""


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
    loss_func = ['categorical_crossentropy', 'categorical_crossentropy']
    print("The COMPILE metric is: " + str(['categorical_crossentropy']))
    age_weights = {0: 1, 1: 1.8446, 2: 0.5949, 3: 0.3593, 4: 0.61249, 5: 1.2032, 6: 1.2199, 7: 3.39511, 8: 3.3142,
                   9: 8.4695}
    gen_weights = {0: 0.955, 1: 1.049}
    # model.compile(loss={NET_TYPE+'_GenderOut':loss_gender, NET_TYPE+'_AgeOut':loss_age}, optimizer=opt, loss_weights=[1, 1], metrics={NET_TYPE+'_GenderOut':'accuracy', NET_TYPE+'_AgeOut':'accuracy'})
    # model.compile(loss={NET_TYPE+'_GenderOut':loss_gender, NET_TYPE+'_AgeOut':loss_age}, optimizer=opt, loss_weights=[gen_weights, age_weights], metrics={NET_TYPE+'_GenderOut':'accuracy', NET_TYPE+'_AgeOut':'accuracy'})
    model.compile(loss=loss_func, optimizer=opt, loss_weights=[gen_weights, age_weights],
                  metrics={NET_TYPE + '_GenderOut': 'accuracy', NET_TYPE + '_AgeOut': 'accuracy'})
    # matriz de confusion
    print(model.summary())
    return model


"""# Custom Image Data Generator"""


class CustomImageDataGenerator(object):
    def __init__(self, data_type, flip_augm=False):
        self.reset()
        self.data_type = data_type
        self.flip_augm = flip_augm

    def reset(self):
        self.images = []
        self.labels = [[], []]

    def read_images(self, directory, max_paths=15, with_prints=False):
        image_paths = []
        invalids = []
        images_read = 0
        for paths in pathlib.Path(directory).iterdir():
            print("THE PATH " + str(paths)) if with_prints else None
            for path in pathlib.Path(str(paths)).iterdir():
                try:
                    images_read += 1
                    print("IMAGES: " + str(images_read)) if with_prints else None
                    if self.data_type == "train" and images_read % TRAIN_VAL_RATIO == 0:
                        continue
                    elif self.data_type == "validation" and images_read % TRAIN_VAL_RATIO != 0:
                        continue
                    gen_label = LABELS_GENDER_SMALL.index(path.stem.split('_')[0])
                    age_label = LABELS_AGE.index(path.stem.split('_')[1])
                    # if TOTALS["genders"][0] / TOTALS["genders"][1] > 1.5:
                    TOTALS["genders"][gen_label] += 1
                    TOTALS["ages"][age_label] += 1
                    image_paths.append(path)
                except Exception as e:
                    invalids.append(path)
                    # print("\nError with image: " + str(e))
                    # print("Image path: " + str(path))
        if with_prints:
            print("Invalids: " + str(invalids))
            print(self.data_type + " total: " + str(len(image_paths)))
        return image_paths

    # flow o flow_from_directory?
    def flow(self, directory, classes, batch_size=32):
        first = True
        shuffled_paths = []
        while True:
            paths_to_remove = []
            if len(shuffled_paths) < batch_size:
                # print("\nType: "+self.data_type+". Reading all again (Images left: "+ str(len(shuffled_paths))+")")
                shuffled_paths = self.read_images(directory)
                shuffle(shuffled_paths)
                # print("New size: "+ str(len(shuffled_paths)))
            for path in shuffled_paths:
                paths_to_remove.append(path)
                try:
                    gen_label = LABELS_GENDER_SMALL.index(path.stem.split('_')[0])
                    age_label = LABELS_AGE.index(path.stem.split('_')[1])
                    with pil_image.open(path) as f:
                        image_raw = np.asarray(f.convert('RGB'), dtype=np.float32)
                        image = cv2.resize(image_raw, (IMAGE_WIDTH, IMAGE_HEIGHT))
                        self.images.append(image)
                        self.labels[0].append(gen_label)
                        self.labels[1].append(age_label)
                        if self.flip_augm:
                            self.images.append(
                                cv2.flip(image, 1))  # 1 -> horizontal flip, 0 -> vertical flip, -1 -> both axis flip
                            self.labels[0].append(gen_label)
                            self.labels[1].append(age_label)
                        # self.images.append(np.asarray(f, dtype=np.float32))
                except Exception as e:
                    print("\nError in IDG: " + str(e))
                    print("Image path: " + str(path))
                    continue
                if len(self.images) == batch_size:
                    break
            if first:
                print_batch(self.images, self.labels[0], self.labels[1])
                first = False
            inputs = np.asarray(self.images, dtype=np.float32)
            # inputs = tuple([np.stack(samples, axis=0) for samples in zip(*self.images)])
            # inputs = tuple(inputs)
            targets = {
                NET_TYPE + '_GenderOut': np.asarray(to_categorical(self.labels[0], num_classes=2), dtype=np.float32),
                NET_TYPE + '_AgeOut': np.asarray(to_categorical(self.labels[1], num_classes=10), dtype=np.float32)}
            # targets = [np.asarray(to_categorical(self.labels[0], num_classes=2), dtype=np.float32),np.asarray(to_categorical(self.labels[1], num_classes=10), dtype=np.float32)]
            self.reset()
            # print(inputs.shape)
            for path_remove in paths_to_remove:
                shuffled_paths.remove(path_remove)
            # print("\nShuffled lenght: " + str(len(shuffled_paths)) + " | Directory: " + str(directory))
            # print("INPUTS: " + str(inputs.shape))
            # print("TARGETS GENDER: " + str(targets['GenderOut'].shape))
            # print("TARGETS AGE: " + str(targets['AgeOut'].shape))
            if NET_TYPE == None and len(shuffled_paths) == 0:
                print("returning because nothing")
                return
            else:
                yield (inputs, targets)


"""# Prediction and train functions"""

# Comparar VGG_age_gender vs VGG_AGE Y VGG_GENDER
#### TRAINING PARALEL FEATURE PROCESSING NETWORK ###
COMMON_MODEL = None
PARALEL_MODEL = None
NDDR_MODEL = None
SEPARATED_MODEL = None

global RESULTS_FILENAME, RESULTS_DICT
RESULTS_FILENAME = create_results_filename()
RESULTS_DICT = {'Headers': create_header()}


def do_the_run(epochs=EPOCHS):
    # 'Epchos','Train steps','Val steps','Batch Size'
    global NET_TYPE
    NET_TYPE = "VGG16_COMMON"
    print(RESULTS_DICT)
    print("EPOCHS::: " + str(EPOCHS))
    RESULTS_DICT[NET_TYPE] = [NET_TYPE, str(epochs), str(STEPS_EPOCHS), str(STEPS_VAL), str(BATCH_SIZE)]
    print("\nStarting training with net " + str(NET_TYPE))

    model = build_vgg16_net()

    history = train_model(model, DATA_PATH, DATA_PATH, net_type, epochs)
    plot_history(history)
    RESULTS_DICT[NET_TYPE].append(str(model.count_params()))
    print("\nAmount of params: " + str(model.count_params()))
    # save_model(model)
    # predict_images(model) This should NOT go here. But it's useful to compare model vs loaded model (to valdiate save is working correctly)

    # RESULTS_DICT[NET_TYPE].extend([str(history.history['val_'+NET_TYPE+'_GenderOut_accuracy'][-1]), str(history.history['val_'+NET_TYPE+'_GenderOut_accuracy'][-1]), str(history.history['val_'+NET_TYPE+'_AgeOut_accuracy'][-1]), str(history.history['val_'+NET_TYPE+'_AgeOut_loss'][-1])])
    # print("val_"+NET _TYPE+"_GenderOut_accuracy: "+str(history.history['val_'+NET_TYPE+'_GenderOut_accuracy'])); print("val_"+NET_TYPE+"_AgeOut_accuracy: "+str(history.history['val_'+NET_TYPE+'_AgeOut_accuracy'])); print("val_"+NET_TYPE+"_GenderOut_loss: "+str(history.history['val_'+NET_TYPE+'_GenderOut_loss'])); print("val_"+NET_TYPE+"_AgeOut_loss: "+str(history.history['val_'+NET_TYPE+'_AgeOut_loss'])); print("combined loss: "+str(history.history['loss']))
    RESULTS_DICT[NET_TYPE].append(str(history.history['loss'][-1]))
    return model


def do_predictions(epochs=-1):
    RESULTS_DICT[NET_TYPE] = [NET_TYPE, str(epochs), str(STEPS_EPOCHS), str(STEPS_VAL), str(BATCH_SIZE)]
    print("\nStarting load & predict with net " + str(NET_TYPE))
    model = load_model_from_disk()
    if epochs > 0:
        history = train_model(model, DATA_PATH, DATA_PATH, net_type, epochs)
        plot_history(history)
        save_model(model)
    predict_images(model)
    return model


drive.mount('/content/gdrive')  # , force_remount=True)
print(keras.__version__)
global MODEL_NDDR, MODEL_COMMON, MODEL_PARALEL, MODEL_SEPARATED, MODEL_SEPARATED_GENDER, MODEL_SEPARATED_AGE, DO_TRAIN
DO_TRAIN = True
train_images_data_gen = CustomImageDataGenerator("train")
validation_images_data_gen = CustomImageDataGenerator("validation")
print("TRAINING IMAGES: Required: " + str(BATCH_SIZE * EPOCHS * STEPS_EPOCHS) + ", actual: " + str(
    len(train_images_data_gen.read_images(DATA_PATH))))
print("VALIDATION IMAGES: Required: " + str(BATCH_SIZE_VAL * EPOCHS * STEPS_VAL) + ", actual: " + str(
    len(validation_images_data_gen.read_images(DATA_PATH))))
if DO_TRAIN:  # NET_TYPE = "COMMON" - "PARALEL" - "NDDR" - "SEPARATED"
    train_data = train_images_data_gen.flow(DATA_PATH, [], 1)  # B
    valid_data = validation_images_data_gen.flow(DATA_PATH, [], 1)  # B
# else:
#     COMMON_MODEL = do_predictions("COMMON", epochs=10)
#     PARALEL_MODEL = do_predictions("PARALEL", epochs=10)
#     NDDR_MODEL = do_predictions("NDDR", epochs=10)
#     SEPARATED_MODEL = do_predictions("SEPARATED", epochs=10)
#     SEPARATED_GENDER_MODEL = do_predictions("SEPARATED_GENDER", epochs=10)
#     SEPARATED_AGE_MODEL = do_predictions("SEPARATED_AGE", epochs=10)

print(RESULTS_DICT)
write_to_file(create_results_filename())

"""# Train models"""
train = True
predict = True
# for key, value in d.items():
MODEL = None
try:
    if train:
        MODEL = do_the_run()
        save_model(MODEL)
    if predict:
        MODEL = do_predictions()
except Exception as e:
    print("\nError with model VGG16_COMMON: " + str(e))
    traceback.print_exc()

predict_images(MODEL, max_imgs=1500, with_prints=False)


tf.keras.utils.plot_model(MODEL)