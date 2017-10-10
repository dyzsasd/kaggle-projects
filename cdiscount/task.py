from collections import defaultdict
import os
import pickle
import random
import sys

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array, Iterator
import numpy as np
import pymongo as pym

from train.model import get_model
from train.utils import parse_bson_obj, execute

input_shape = (180, 180)


class MongoImageIterator(Iterator):
    def __init__(self, path, batch_size, input_shape):
        self.cli = pym.MongoClient('mongodb://localhost/test')
        self.db = self.cli['test']
        self.col = self.db['train']

        self.path = path
        self.input_shape = input_shape

        id2cat = pickle.load(open(os.path.join(self.path, 'id2cat.pk'), 'rb'))
        self.cat2vec = pickle.load(
            open(os.path.join(self.path, 'cat2vec.pk'), 'rb'))

        self.train_set, self.validation_set = self.separate_dataset(id2cat)
        super(MongoImageIterator, self).__init__(
            len(self.train_set), batch_size, True, None)

    def load_validation_dataset(self):
        print("loading %s validation samples" % len(self.validation_set))
        self.validation_x, self.validation_y = self.get_batch_samples(
            self.validation_set, pos=10000, neg=20000)

    def separate_dataset(self, image_metas):
        category_ids = defaultdict(list)
        for image_id, image_cat in image_metas.items():
            category_ids[image_cat].append(image_id)
        train_set = {}
        validation_set = {}
        for cat, ids in category_ids.items():
            random.shuffle(ids)
            sep = len(ids) // 100
            train_set[cat] = ids[sep:]
            validation_set[cat] = ids[:sep]

        return train_set, validation_set

    def get_batch_samples(self, cat_ids, pos=50, neg=50):
        batch = []
        pos_per_cat = max(pos // len(cat_ids), 1)

        cat_images = {}
        for cat, ids in cat_ids.items():
            images = []
            for obj in self.col.find({'_id': {'$in': ids}}):
                obj = parse_bson_obj(obj)
                images.extend(obj['imgs'])
            if len(images) > 1:
                cat_images[cat] = images

            if len(images) < 2 or pos < 0:
                continue

            tuples = zip(
                random.sample(images, min(len(images, pos_per_cat))),
                random.sample(images, min(len(images, pos_per_cat)))
            )
            batch.extends(zip(tuples, [1] * len(tuples)))
            pos = pos - len(tuples)

        if len(cat_images) > 2:
            max_neg = (len(cat_images) ** 2)
            for i in range(min(neg, max_neg)):
                selected_cats = random.sample(cat_images.keys(), 2)
                batch.append(
                    (
                        (
                            random.sample(cat_images[selected_cats[0]], 1)[0],
                            random.sample(cat_images[selected_cats[1]], 1)[0],
                        ),
                        0
                    )
                )

        batch_x_1 = np.zeros((len(batch), ) + self.input_shape + (3, ))
        batch_x_2 = np.zeros((len(batch), ) + self.input_shape + (3, ))

        count = 0
        for tup, _ in batch:
            image_1, image_2 = tup
            x_1 = img_to_array(image_1)
            x_2 = img_to_array(image_2)
            batch_x_1[count] = x_1
            batch_x_2[count] = x_2
            count += 1
        batch_y = self.cat2vec.transform([tup[-1] for tup in batch])

        return ((batch_x_1, batch_x_2), batch_y)

    def _get_image_tuples(self, id_cat_tuples):
        chunk_size = 100

        chunks = []
        for index in range(0, len(id_cat_tuples), chunk_size):
            chunks.append((
                [tup[0] for tup in id_cat_tuples[index: index + chunk_size]],
            ))

        resps, _ = execute(self.get_batch_samples, chunks)

        batch_x = np.concatenate([tup[0] for tup in resps])
        batch_y = np.concatenate([tup[1] for tup in resps])
        return batch_x, batch_y

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        items = self.train_set()
        return self.get_batch_samples(dict([
            items[chosen_index] for chosen_index in index_array
        ]))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'val':
        validation = True
    else:
        validation = False

    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    callbacks_list = [checkpoint]

    generator = MongoImageIterator('/data/cdiscount', 1, input_shape)
    model = get_model(
        input_shape + (3, ),
        len(generator.cat2vec.classes_),
        pool_size=5,
        strides=3
    )

    if validation:
        generator.load_validation_dataset()
        model.fit_generator(
            generator,
            steps_per_epoch=1,
            epochs=10000,
            validation_data=[generator.validation_x, generator.validation_y],
            callbacks=callbacks_list
        )
    else:
        model.fit_generator(
            generator,
            steps_per_epoch=100,
            epochs=10000,
            callbacks=callbacks_list
        )
