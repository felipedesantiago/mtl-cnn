"""# Auxiliar Functions"""
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from IPython.display import Image, display
import os, cv2, csv, datetime, numpy as np
import comet_ml
from parameters import *

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


def plot_history(history, net_type=VGG16_COMMON):
    # {'loss': [2.7617995738983154], 'GenderOut_loss': [0.6187705993652344], 'AgeOut_loss': [2.143028974533081], 'GenderOut_accuracy': [0.7515624761581421], 'AgeOut_accuracy': [0.2515625059604645]}
    # Agregar validación a las gráficas
    # val_loss - val_GenderOut_loss - val_AgeOut_loss - val_GenderOut_accuracy - val_AgeOut_accuracy

    # print(d["thekey"]) if "the key" in d else None
    print("The history: " + str(history.history))
    plt.style.use('seaborn-whitegrid')
    plt.axes()
    plt.plot(history.history['val_' + net_type + '_GenderOut_accuracy'], 'r',
             label="GenderVAL") if 'val_' + net_type + '_GenderOut_accuracy' in history.history else None
    # plt.plot(history.history[net_type+'_GenderOut_accuracy'],'m', label="GenderTRAIN") if net_type+'_GenderOut_accuracy' in history.history else None
    plt.plot(history.history['val_' + net_type + '_AgeOut_accuracy'], 'b',
             label="AgeVAL") if 'val_' + net_type + '_AgeOut_accuracy' in history.history else None
    # plt.plot(history.history[net_type+'_AgeOut_accuracy'],'c', label="AgeTRAIN") if net_type+'_AgeOut_accuracy' in history.history else None
    plt.plot(history.history['accuracy'], 'm', label="TRAIN") if 'accuracy' in history.history else None
    plt.plot(history.history['val_accuracy'], 'r', label="VAL") if 'val_accuracy' in history.history else None
    plt.title('Accuracy ' + net_type)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    # summarize history for loss
    plt.style.use('seaborn-whitegrid')
    plt.axes()
    plt.plot(history.history['loss'], 'g', label="Combined") if 'loss' in history.history else None
    plt.plot(history.history['val_' + net_type + '_GenderOut_loss'], 'r',
             label="GenderVAL") if 'val_' + net_type + '_GenderOut_loss' in history.history else None
    # plt.plot(history.history[net_type+'_GenderOut_loss'],'m', label="GenderTRAIN") if net_type+'_GenderOut_loss' in history.history else None
    plt.plot(history.history['val_' + net_type + '_AgeOut_loss'], 'b',
             label="AgeVAL") if 'val_' + net_type + '_AgeOut_loss' in history.history else None
    # plt.plot(history.history[net_type+'_AgeOut_loss'],'c', label="AgeTRAIN") if net_type+'_AgeOut_loss' in history.history else None
    plt.plot(history.history['loss'], 'm', label="TRAIN") if 'loss' in history.history else None
    plt.plot(history.history['val_loss'], 'r', label="VAL") if 'val_loss' in history.history else None
    plt.title('Loss ' + net_type)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def predict_images(model, max_imgs=MAX_PREDICTION_IMAGES, with_prints=False, net_type=VGG16_COMMON):
    preview_size = 50
    print("Predicting...")
    print(TOTALS)
    predicted = 0
    pred_accuracy = {"gender": [0, 0], "age": [0, 0]}
    gen_idx = 0
    age_idx = 1
    # if "SEPARATED_AGE" in net_type:
    #     age_idx = 0
    start_time = datetime.datetime.now()
    y_true = [[], []]
    y_pred = [[], []]
    for img_path in os.listdir(DATA_TEST_PATH):
        # TRY to do a model.predict() with a set or list of images, instead of one by one?
        if with_prints:
            print("\nImage: " + str(img_path))
            display(Image(os.path.join(DATA_TEST_PATH, img_path), width=preview_size, height=preview_size))

        img = cv2.imread(os.path.join(DATA_TEST_PATH, img_path))
        # img = img[..., ::-1]
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = np.expand_dims(img, axis=0)
        # print("predicting the net: "+net_type)
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
    print("Gender error average: " + str(pred_accuracy["gender"][0] / pred_accuracy["gender"][1]))
    print("Age error average: " + str(pred_accuracy["age"][0] / pred_accuracy["age"][1]))
    # RESULTS_DICT[net_type].append(str(diff_secs / max_imgs))
    # RESULTS_DICT[net_type].append(str(pred_accuracy["gender"][0] / pred_accuracy["gender"][1])) if "SEPARATED_AGE" not in net_type else \
    # RESULTS_DICT[net_type].append("")
    # RESULTS_DICT[net_type].append(str(pred_accuracy["age"][0] / pred_accuracy["age"][1])) if "SEPARATED_GENDER" not in net_type else RESULTS_DICT[net_type].append("")
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
    experiment.log_confusion_matrix(y_true[0], y_pred[0], title=net_type + " Gender", file_name=net_type + "_Gender.json", labels=["Male", "Female"])
    experiment.log_confusion_matrix(mapped_true, mapped_pred, title=net_type + " Age", file_name=net_type + "_Age.json") # , labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", ">90"])
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
        # for key in RESULTS_DICT.keys():
        #     writer.writerow(RESULTS_DICT[key])
        # csvfile.close()
    print("Updated results in: " + str(filename))