"""# Train Models"""
# https://towardsdatascience.com/dealing-with-imbalanced-data-in-tensorflow-class-weights-60f876911f99
# https://ichi.pro/es/manejo-de-datos-desequilibrados-en-tensorflow-ponderaciones-de-clase-106620257378201
# https://gist.github.com/angeligareta/83d9024c5e72ac9ebc34c9f0b073c64c

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
from custom_image_data_generator import CustomImageDataGenerator
from keras.callbacks import ModelCheckpoint
import datetime
from parameters import *



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


# Method by Morten Grøftehauge. # Source: https://github.com/keras-team/keras/issues/11735#issuecomment-641775516

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
    #with EXPERIMENT.train():
    history = model.fit_generator(generator=train_data, validation_data=valid_data, validation_steps=STEPS_VAL, steps_per_epoch=STEPS_EPOCHS, epochs=epochs, callbacks=[checkpoint], verbose=1)
    diff_secs = (datetime.datetime.now() - start_time).total_seconds()
    RESULTS_DICT[NET_TYPE].append(str(diff_secs / epochs))
    print("Total TRAIN time: " + str(diff_secs) + " - AVG: " + str(diff_secs) + "/" + str(epochs) + " = " + str(
        diff_secs / epochs))
    return history