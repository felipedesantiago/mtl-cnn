"""# Custom Image Data Generator"""
import pathlib, cv2, numpy as np
from PIL import Image as pil_image
from random import shuffle
from utils import print_batch
from tensorflow.keras.utils import to_categorical
from parameters import *

class CustomImageDataGenerator(object):
    def __init__(self, data_type, flip_augm=False):
        self.reset()
        self.data_type = data_type
        self.flip_augm = flip_augm

    def reset(self):
        self.images = []
        self.labels = [[], []]

    def read_images(self, directory, with_prints=False):
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
    def flow(self, directory, net_type, batch_size=32):
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
                net_type + '_GenderOut': np.asarray(to_categorical(self.labels[0], num_classes=GENDER_CLASSES), dtype=np.float32),
                net_type + '_AgeOut': np.asarray(to_categorical(self.labels[1], num_classes=AGE_CLASSES), dtype=np.float32)}
            # targets = [np.asarray(to_categorical(self.labels[0], num_classes=2), dtype=np.float32),np.asarray(to_categorical(self.labels[1], num_classes=10), dtype=np.float32)]
            self.reset()
            # print(inputs.shape)
            for path_remove in paths_to_remove:
                shuffled_paths.remove(path_remove)
            # print("\nShuffled lenght: " + str(len(shuffled_paths)) + " | Directory: " + str(directory))
            # print("INPUTS: " + str(inputs.shape))
            # print("TARGETS GENDER: " + str(targets['GenderOut'].shape))
            # print("TARGETS AGE: " + str(targets['AgeOut'].shape))
            if net_type is None and len(shuffled_paths) == 0:
                print("returning because nothing")
                return
            elif net_type == VGG16_INDEPENDENT:
                yield inputs, targets
            else:
                yield [inputs, inputs], targets