"""# Load & Save Model"""
from keras.models import load_model
from parameters import *

def save_model(my_model, net_type=VGG16_COMMON):
    print("SAVE DATA:")
    print(str(MODEL_PATH))
    print(str(net_type))
    my_model.save(filepath=MODEL_PATH + net_type + "2_model", overwrite=True)
    print("Saved model: " + MODEL_PATH + net_type + "2_model")


def load_model_from_disk(net_type=NET_TYPE):
    print("Loading model: " + str(MODEL_PATH) + str(net_type) + "2_model")
    to_ret = load_model(filepath=MODEL_PATH + net_type + "2_model")
    print("Loaded model: " + str(MODEL_PATH) + str(net_type) + "2_model")
    return to_ret