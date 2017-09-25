import matplotlib

matplotlib.use('agg')

import fnmatch
import os
import random

from keras.callbacks import ModelCheckpoint
from keras.metrics import binary_accuracy
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, load_img, Iterator
import numpy as np

from res_model import get_model
from utils import binary_crossentropy_with_logits

input_shape = (160, 240)

train_path = '/data/car_section/train/'
mask_path = '/data/car_section/train_masks'


class SegDirectoryIterator(Iterator):
    def __init__(self, train_image_path, mask_file, batch_size=19):
        self.train_image_path = train_image_path
        self.masks = self._get_masks(mask_file)
        files = list(self.masks.keys())

        self.train_images = []
        self.validation_images = []

        for file in files:
            if random.random() > 0.1:
                self.train_images.append(file)
            else:
                self.validation_images.append(file)

        self.validation_data = np.zeros(
            (len(self.validation_images), ) + input_shape + (3, ))
        self.validation_mask_data = np.zeros(
            (len(self.validation_images), ) + input_shape + (1, ))

        self.train_data = [None] * len(self.train_images)
        self.mask_data = [None] * len(self.train_images)

        cursor = 0
        for data_file in self.validation_images:
            self.validation_data[cursor] = self._load_image(
                os.path.join(self.train_image_path, data_file))
            self.validation_mask_data[cursor] = self.masks[data_file]
            cursor += 1

        super(SegDirectoryIterator, self).__init__(
            len(self.train_images), batch_size, True, 1)

    def _load_image(self, image_path):
        img_x = load_img(image_path)
        x = img_to_array(img_x)
        x = x / 255.
        return x

    def _mask_to_matrix(self, mask, width, height):
        data = np.zeros()
        for i in range(0, len(mask), 2):
            data[mask[i]: mask[i] + mask[i + 1]] = 1.
        return data.reshape((height, width))

    def _get_masks(self, mask_file, width, height):
        lines = open(mask_file, 'r').readlines()
        masks = {
            items[0]: self._mask_to_matrix(
                [int(num) for num in items[1].split(" ")], width, height)
            for items in [
                line.split(",") for line in lines[1:]]
        }
        return masks

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)

        # The transformation of images is not under thread lock so it can be
        # done in parallel
        batch_x = np.zeros((current_batch_size, ) + input_shape + (3, ))
        batch_y = np.zeros((current_batch_size, ) + input_shape + (1, ))

        for i, j in enumerate(index_array):
            if self.train_data[j] is None:
                data_file = self.train_images[j]
                self.train_data[j] = self._load_image(
                    os.path.join(self.train_image_path, data_file))
                self.mask_data[j] = self.masks[data_file]

            batch_x[i] = self.train_data[j]
            batch_y[i] = self.mask_data[j]

        return (batch_x, batch_y)


def get_checkpoint_callback(model_dir):
    filepath = model_dir + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1,
        save_best_only=True, mode='max')
    return checkpoint


callbacks_list = [get_checkpoint_callback("")]


model = get_model(input_shape + (3, ))

# model.fit_generator(SegDirectoryIterator(), steps_per_epoch=1000, epochs=10)
sdi = SegDirectoryIterator()
model.fit_generator(
    sdi,
    steps_per_epoch=10,
    epochs=200,
    validation_data=[sdi.validation_data, sdi.validation_mask_data],
    callbacks=callbacks_list
)
# model.fit(sdi.self.train_images_data, sdi.self.mask_images_data, batch_size=100, epochs=100)
