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
EPOCHS = 3 # 40
STEPS_EPOCHS = 10 # 1182  # training images #trainimages / batch_size * 2 (2 because of DA flip)
STEPS_VAL = 3 # 294  # valid images #trainimages / batch_size * 2 (2 because of DA flip)

DATA_PATH = "/content/gdrive/MyDrive/ColabNotebooks/images/datasets/UTKFace/"
DATA_TEST_PATH = "/content/gdrive/MyDrive/ColabNotebooks/images/datasets/predict/"
MODEL_PATH = "/content/gdrive/MyDrive/ColabNotebooks/models/"
DO_TRAIN = True
PLOT_MODELS = True
NET_TYPE = None

GENDER_CLASSES = 2
AGE_CLASSES = 7

LABELS_GENDER = ['male', 'female']
LABELS_GENDER_SMALL = ['m', 'f']
# LABELS_AGE = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']
LABELS_AGE = ['0', '10', '20', '30', '40', '50', '70']

# TOTALS = {"genders": {0: 0, 1: 0}, "ages": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}}
TOTALS = {"genders": {0: 0, 1: 0}, "ages": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}}
AGE_CLASS_INDEX = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:5, 7:6, 8:6, 9:6}

MAX_PREDICTION_IMAGES = 1000  # 75 #MAX: 78

GEN_WEIGHTS = {0: 0.955, 1: 1.049}
# AGE_WEIGHTS = {0: 1, 1: 1.8446, 2: 0.5949, 3: 0.3593, 4: 0.61249, 5: 1.2032, 6: 1.2199, 7: 3.39511, 8: 3.3142, 9: 8.4695}
AGE_WEIGHTS = {0:1,1:1.8446,2:0.5949,3:0.3593,4:0.61249,5:1.2032,6:1.4000}

# NET TYPES
VGG16_COMMON = "VGG16_COMMON"
VGG16_INDEPENDENT = "VGG16_INDEPENDENT"
VGG16_NDDR = "VGG16_NDDR"
MN_COMMON = "MN_COMMON"
MN_INDEPENDENT = "MN_INDEPENDENT"
MN_NDDR = "MN_NDDR"
