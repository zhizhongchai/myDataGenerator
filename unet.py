# coding: utf-8
'''
    - "ZF_UNET_224" Model based on UNET code from following paper: https://arxiv.org/abs/1505.04597
    - This model used to get 2nd place in DSTL competition: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
    - For training used DICE coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    - Input shape for model is 224x224 (the same as for other popular CNNs like VGG or ResNet)
    - It has 3 input channels (to process standard RGB (BGR) images). You can change it with variable "INPUT_CHANNELS"
    - It trained on random image generator with random light shapes (ellipses) on dark background with noise (< 10%).
    - In most cases model ZF_UNET_224 is ok to be used without pretrained weights.
'''

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras import backend as K
from keras.layers.merge import concatenate

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1


def preprocess_batch(batch):
    batch /= 256
    batch -= 0.5
    return batch


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def double_conv_layer(x, size, dropout, batch_norm):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv


def UNET(input_shape, dropout_val=0.2, batch_norm=True):
    inputs = Input(input_shape)
    filters = 32
    axis = -1

    conv_224 = double_conv_layer(inputs, filters, 0, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2 * filters, 0, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4 * filters, 0, batch_norm)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8 * filters, 0, batch_norm)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16 * filters, 0, batch_norm)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32 * filters, 0, batch_norm)
    output1 = BatchNormalization(axis=axis, epsilon=1.1e-5)(conv_7)
    output1 = Activation('relu')(output1)
    output1 = GlobalAveragePooling2D()(output1)
    output1 = Dense(2, activation='sigmoid')(output1)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16 * filters, 0, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8 * filters, 0, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4 * filters, 0, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2 * filters, 0, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val, batch_norm)

    conv_final = Conv2D(3, (1, 1))(up_conv_224)
    # conv_final = Activation('sigmoid')(conv_final)
    conv_final = Conv2D(1, 1, activation='sigmoid')(conv_final)

    model = Model(inputs, outputs=[output1, conv_final], name="UNET_320")
    return model


if __name__ == '__main__':
    from keras.optimizers import RMSprop

    input_shape = (320, 320, 3)
    classes = 3
    model_savepath = 'models/unet_model_1_1_320_0423/'

    model = UNET(input_shape=input_shape)

    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0001)

    model.summary()
