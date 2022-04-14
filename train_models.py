"""# Train Models"""
# https://towardsdatascience.com/dealing-with-imbalanced-data-in-tensorflow-class-weights-60f876911f99
# https://ichi.pro/es/manejo-de-datos-desequilibrados-en-tensorflow-ponderaciones-de-clase-106620257378201
# https://gist.github.com/angeligareta/83d9024c5e72ac9ebc34c9f0b073c64c

from custom_image_data_generator import CustomImageDataGenerator
from keras.callbacks import ModelCheckpoint
import datetime
from parameters import *

def train_model(model, train_path, validation_path, epochs=EPOCHS):  # MAL EL CLASS MODE
    train_data_gen = CustomImageDataGenerator("train", flip_augm=True)  # B
    valid_data_gen = CustomImageDataGenerator("validation", flip_augm=True)  # B
    train_data = train_data_gen.flow(train_path, BATCH_SIZE)  # B
    valid_data = valid_data_gen.flow(validation_path, BATCH_SIZE_VAL)  # B
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
    # RESULTS_DICT[NET_TYPE].append(str(diff_secs / epochs))
    print("Total TRAIN time: " + str(diff_secs) + " - AVG: " + str(diff_secs) + "/" + str(epochs) + " = " + str(diff_secs / epochs))
    return history