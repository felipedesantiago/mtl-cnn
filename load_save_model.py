"""# Load & Save Model"""
from keras.models import load_model

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