from pylab import *
from keras.models import Model
from keras.layers import *
from keras.applications.vgg16 import *
from keras.models import *
import keras.backend as K


def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1, 2), keepdims=True)
    std = K.std(tensor, axis=(1, 2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)

    return mvn


def crop(tensors):
    '''
    List of 2 tensors, the second tensor having larger spatial dimensions.
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape(t)
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (crop_h // 2, crop_h // 2 + rem_h)
    crop_w_dims = (crop_w // 2, crop_w // 2 + rem_w)
    cropped = Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1])

    return cropped


def dice_coef(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient per batch.'''
    axes = (1, 2, 3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)


def jaccard_coef(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1, 2, 3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


def get_model(
    input_shape,
    num_classes=2,
    pool_size=2,
    strides=2,
):
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)

    if isinstance(strides, int):
        strides = (strides, strides)

    kwargs = dict(
        kernel_size=3,
        strides=1,
        activation='relu',
        padding='valid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )

    layers = [
        Lambda(mvn, name='mvn0'),
        Conv2D(filters=64, name='conv1', **kwargs),
        Lambda(mvn, name='mvn1'),
        Conv2D(filters=64, name='conv2', **kwargs),
        Lambda(mvn, name='mvn2'),
        Conv2D(filters=64, name='conv3', **kwargs),
        Lambda(mvn, name='mvn3'),
        MaxPooling2D(pool_size=pool_size, strides=strides, padding='valid', name='pool1'),
        Dropout(rate=0.5, name='drop1'),
        Conv2D(filters=128, name='conv4', **kwargs),
        Lambda(mvn, name='mvn4'),
        Conv2D(filters=128, name='conv5', **kwargs),
        Lambda(mvn, name='mvn5'),
        Conv2D(filters=128, name='conv6', **kwargs),
        Lambda(mvn, name='mvn6'),
        Conv2D(filters=128, name='conv7', **kwargs),
        Lambda(mvn, name='mvn7'),
        MaxPooling2D(pool_size=pool_size, strides=strides, padding='valid', name='pool2'),
        Dropout(rate=0.5, name='drop2'),
        Conv2D(filters=256, name='conv8', **kwargs),
        Lambda(mvn, name='mvn8'),
        Conv2D(filters=256, name='conv9', **kwargs),
        Lambda(mvn, name='mvn9'),
        Conv2D(filters=256, name='conv10', **kwargs),
        Lambda(mvn, name='mvn10'),
        Conv2D(filters=256, name='conv11', **kwargs),
        Lambda(mvn, name='mvn11'),
        MaxPooling2D(pool_size=pool_size, strides=strides, padding='valid', name='pool3'),
        Dropout(rate=0.5, name='drop3'),
        Conv2D(filters=512, name='conv12', **kwargs),
        Lambda(mvn, name='mvn12'),
        Conv2D(filters=512, name='conv13', **kwargs),
        Lambda(mvn, name='mvn13'),
        Conv2D(filters=512, name='conv14', **kwargs),
        Lambda(mvn, name='mvn14'),
        Conv2D(filters=512, name='conv15', **kwargs),
        Lambda(mvn, name='mvn15'),
        Flatten(),
        Dense(num_classes, activation='relu'),
    ]

    input_1 = Input(shape=input_shape, dtype='float', name='data')
    input_2 = Input(shape=input_shape, dtype='float', name='data')

    output_1 = input_1
    for layer in layers:
        output_1 = layer(output_1)

    output_2 = input_2
    for layer in layers:
        output_2 = layer(output_2)

    merged_vector = concatenate([output_1, output_2], axis=-1)
    output = Dense(1, activation="sigmoid")(merged_vector)
    model = Model([input_1, input_2], output)

    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
