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
        self.validation_x, self.validation_y = self._get_image_tuples(
            self.validation_set)

    def separate_dataset(self, image_metas):
        category_ids = defaultdict(list)
        for image_id, image_cat in image_metas.items():
            category_ids[image_cat].append(image_id)
        train_set = []
        validation_set = []
        for cat, ids in category_ids.items():
            random.shuffle(ids)
            sep = len(ids) // 100
            train_set.extend([(_id, image_metas[_id]) for _id in ids[sep:]])
            validation_set.extend([
                (_id, image_metas[_id]) for _id in ids[:sep]])

        return train_set, validation_set

    def get_batch_samples(self, ids):
        images = []
        cats = []
        for obj in self.col.find({'_id': {'$in': ids}}):
            obj = parse_bson_obj(obj)
            cats.extend([obj['category_id']] * len(obj['imgs']))
            images.extend(obj['imgs'])
        batch_x = np.zeros((len(images), ) + self.input_shape + (3, ))
        batch_y = self.cat2vec.transform(cats)
        count = 0
        for image in images:
            x = img_to_array(image)
            batch_x[count] = x
            count += 1
        print(batch_x.shape)
        return batch_x, batch_y

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
        return self._get_image_tuples([
            self.train_set[chosen_index] for chosen_index in index_array
        ])


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
