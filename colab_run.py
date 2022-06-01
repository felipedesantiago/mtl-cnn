# -*- coding: utf-8 -*-
"""VGG16 Common.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mg240dE2EJ9Ujwn5VEIqPuf_-L_aqy6G

# Installs & Imports
"""

# datastes: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
import tensorflow as tf, traceback
from tensorflow import keras as keras

import parameters
from custom_image_data_generator import CustomImageDataGenerator
from utils import create_results_filename, create_header, write_to_file
from run_predict import do_predictions, do_the_run, predict_images
from load_save_model import save_model

from parameters import *


global RESULTS_FILENAME, RESULTS_DICT
RESULTS_FILENAME = create_results_filename()
RESULTS_DICT = {'Headers': create_header()}

print(keras.__version__)
DO_TRAIN = True
def test_generator(net_type):
    train_images_data_gen = CustomImageDataGenerator("train")
    validation_images_data_gen = CustomImageDataGenerator("validation")
    print("TRAINING IMAGES: Required: " + str(BATCH_SIZE * EPOCHS * STEPS_EPOCHS) + ", actual: " + str(len(train_images_data_gen.read_images(DATA_PATH))))
    print("VALIDATION IMAGES: Required: " + str(BATCH_SIZE_VAL * EPOCHS * STEPS_VAL) + ", actual: " + str(len(validation_images_data_gen.read_images(DATA_PATH))))
    if DO_TRAIN:
        train_data = train_images_data_gen.flow(DATA_PATH, net_type, 1)  # B
        valid_data = validation_images_data_gen.flow(DATA_PATH, net_type, 1)  # B

    # print(RESULTS_DICT)

def train_and_predict(train=True, predict=True, net_type="VGG16_COMMON", test=False):
    # for key, value in d.items():
    model = None
    parameters.TEST = test
    try:
        # write_to_file(create_results_filename())
        if train:
            model = do_the_run(net_type=net_type) # LINE TO EDIT IN EACH DIFFERENT NOTEBOOK
            save_model(model, net_type)
        if predict:
            model = do_predictions(net_type=net_type)
        # predict_images(model, max_imgs=1500, with_prints=False, net_type=net_type)
        tf.keras.utils.plot_model(model)
    except Exception as e:
        print("\nError with model "+str(model)+": " + str(e))
        traceback.print_exc()
