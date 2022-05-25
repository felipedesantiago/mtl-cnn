"""# Prediction and train functions"""
# Comparar VGG_age_gender vs VGG_AGE Y VGG_GENDER
#### TRAINING PARALEL FEATURE PROCESSING NETWORK ###
from model import train_model, build_model_net
from custom_losses import get_loss
# from utils import create_header, create_results_filename, matrix_as_shape_list, plot_history, predict_images, print_batch, write_to_file
from utils import plot_history, predict_images
from load_save_model import load_model_from_disk, save_model
from parameters import *

def do_the_run(net_type, test=False):
    # 'Epchos','Train steps','Val steps','Batch Size'
    # global NET_TYPE
    # NET_TYPE = net_type # "VGG16_COMMON"
    # print(RESULTS_DICT)
    print("EPOCHS::: " + str(EPOCHS))
    # RESULTS_DICT[NET_TYPE] = [NET_TYPE, str(epochs), str(STEPS_EPOCHS), str(STEPS_VAL), str(BATCH_SIZE)]
    print("\nStarting training with net " + net_type)

    model = build_model_net(net_type)
    if model is None:
        print("MODEL IS NONE")
    else:
        print("MODEL AINT NONE")

    history = train_model(model, DATA_PATH, DATA_PATH, net_type)
    plot_history(history)
    # RESULTS_DICT[NET_TYPE].append(str(model.count_params()))
    print("\nAmount of params: " + str(model.count_params()))
    # save_model(model)
    # predict_images(model) This should NOT go here. But it's useful to compare model vs loaded model (to valdiate save is working correctly)

    # RESULTS_DICT[NET_TYPE].extend([str(history.history['val_'+NET_TYPE+'_GenderOut_accuracy'][-1]), str(history.history['val_'+NET_TYPE+'_GenderOut_accuracy'][-1]), str(history.history['val_'+NET_TYPE+'_AgeOut_accuracy'][-1]), str(history.history['val_'+NET_TYPE+'_AgeOut_loss'][-1])])
    # print("val_"+NET _TYPE+"_GenderOut_accuracy: "+str(history.history['val_'+NET_TYPE+'_GenderOut_accuracy'])); print("val_"+NET_TYPE+"_AgeOut_accuracy: "+str(history.history['val_'+NET_TYPE+'_AgeOut_accuracy'])); print("val_"+NET_TYPE+"_GenderOut_loss: "+str(history.history['val_'+NET_TYPE+'_GenderOut_loss'])); print("val_"+NET_TYPE+"_AgeOut_loss: "+str(history.history['val_'+NET_TYPE+'_AgeOut_loss'])); print("combined loss: "+str(history.history['loss']))
    # RESULTS_DICT[NET_TYPE].append(str(history.history['loss'][-1]))
    return model


def do_predictions(net_type):
    # RESULTS_DICT[NET_TYPE] = [NET_TYPE, str(epochs), str(STEPS_EPOCHS), str(STEPS_VAL), str(BATCH_SIZE)]
    print("\nStarting load & predict with net " + net_type)
    model = load_model_from_disk(net_type)
    predict_images(model)
    return model