from keras import backend as K
from keras.regularizers import l2
import keras
import logging
from GR import *
logger = logging.getLogger(__name__)
K.set_image_data_format("channels_last")


def se_slice(x, i):
    return x[:, i, :, :, :, :]


def se_slice_psd(x, i):
    return x[:, :, :, :, 0]


def squeeze_1D(input_layer, reduction_ratio=1):

    r = reduction_ratio
    c = keras.backend.int_shape(input_layer)[-1]
    x = input_layer

    y = keras.layers.GlobalAveragePooling1D()(input_layer)
    y = keras.layers.Dense(c // r)(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dense(c)(y)
    y = keras.layers.Activation('sigmoid')(y)
    y = keras.layers.Reshape([1, c])(y)
    y = keras.layers.multiply([x, y])
    return y


def squeeze_2D(input_layer, reduction_ratio=1):

    r = reduction_ratio
    c = keras.backend.int_shape(input_layer)[-1]
    x = input_layer

    y = keras.layers.GlobalAveragePooling2D()(input_layer)
    y = keras.layers.Dense(c // r)(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dense(c)(y)
    y = keras.layers.Activation('sigmoid')(y)
    y = keras.layers.Reshape([1, 1, c])(y)
    y = keras.layers.multiply([x, y])
    return y


def SE_Fusion(input_1D_layer, input_2D_layer, reduction_ratio=1, activation='sigmoid'):

    r = reduction_ratio
    c1d = keras.backend.int_shape(input_1D_layer)[-1]
    c2d = keras.backend.int_shape(input_2D_layer)[-1]
    c = c1d + c2d

    x1d = input_1D_layer
    x2d = input_2D_layer

    # Squeeze
    y1d = keras.layers.GlobalAveragePooling1D()(input_1D_layer)
    y2d = keras.layers.GlobalAveragePooling2D()(input_2D_layer)

    y = keras.layers.concatenate([y1d, y2d])
    y = keras.layers.Dense(c // r)(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dense(c)(y)
    y = keras.layers.Activation(activation)(y)

    y1d = keras.layers.Lambda(lambda x: x[:, :c1d])(y)
    y1d = keras.layers.Reshape([1, c1d])(y1d)
    y1d = keras.layers.multiply([x1d, y1d])
    y1d = keras.layers.Flatten()(y1d)

    y2d = keras.layers.Lambda(lambda x: x[:, c1d:])(y)
    y2d = keras.layers.Reshape([1, 1, c2d])(y2d)
    y2d = keras.layers.multiply([x2d, y2d])
    y2d = keras.layers.Flatten()(y2d)

    y = keras.layers.concatenate([y1d, y2d])
    return y


def deepsleepnet(intput, Fs, time_filters_nums, bn_mom):
    y1 = keras.layers.Conv1D(name='conv1_small', kernel_size=Fs//2, strides=Fs//16, filters=time_filters_nums, padding='same',
                             use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(intput)
    y1 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y1)
    y1 = keras.layers.LeakyReLU()(y1)

    y1 = keras.layers.MaxPooling1D(pool_size=8, strides=8, padding='same')(y1)
    y1 = keras.layers.Dropout(0.5)(y1)

    y1 = keras.layers.Conv1D(name='conv2_small', kernel_size=8, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y1)
    y1 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y1)
    y1 = keras.layers.LeakyReLU()(y1)

    y1 = keras.layers.Conv1D(name='conv3_small', kernel_size=8, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y1)
    y1 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y1)
    y1 = keras.layers.LeakyReLU()(y1)

    y1 = keras.layers.Conv1D(name='conv4_small', kernel_size=8, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y1)
    y1 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y1)
    y1 = keras.layers.LeakyReLU()(y1)
    y1 = keras.layers.MaxPooling1D(pool_size=4, strides=4, padding='same')(y1)

    y2 = keras.layers.Conv1D(name='conv1_big', kernel_size=Fs*4, strides=Fs//2, filters=time_filters_nums, padding='same',
                             use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(intput)
    y2 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y2)
    y2 = keras.layers.LeakyReLU()(y2)

    y2 = keras.layers.MaxPooling1D(pool_size=4, strides=4, padding='same')(y2)
    y2 = keras.layers.Dropout(0.5)(y2)

    y2 = keras.layers.Conv1D(name='conv2_big', kernel_size=6, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y2)
    y2 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y2)
    y2 = keras.layers.LeakyReLU()(y2)

    y2 = keras.layers.Conv1D(name='conv3_big', kernel_size=6, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y2)
    y2 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y2)
    y2 = keras.layers.LeakyReLU()(y2)

    y2 = keras.layers.Conv1D(name='conv4_big', kernel_size=6, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y2)
    y2 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y2)
    y2 = keras.layers.LeakyReLU()(y2)

    y2 = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(y2)
    y = keras.layers.concatenate([y1, y2], axis=1)
    return y

def getModel(
    num_class,
    num_domain,
    seq_len=100,
    width=16,
    height=16,
    use_bias=True,
    bn_mom=0.9,
    times=7680,
    time_filters_nums=64,
    psd_filter_nums=32,
    reduction_ratio=1,
    se_activation='sigmoid',
    lambda_reversal=0.001,
    Fs=128
):
    input_layer = keras.layers.Input(
        name='input_layer_psd', shape=(seq_len, width, height, 1))
    input_psd = keras.layers.Lambda(
        se_slice_psd, arguments={'i': 1})(input_layer)
    input_psd = keras.layers.core.Permute((2, 3, 1))(input_psd)

    x_psd = keras.layers.Conv2D(name='conv1_middle_psd', kernel_size=(1, 1), strides=(1, 1), filters=psd_filter_nums, padding='same',
                                use_bias=use_bias, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_psd)
    x_psd = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                            beta_initializer='zeros', gamma_initializer='ones',
                                            moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                            beta_regularizer=None, gamma_regularizer=None,
                                            beta_constraint=None, gamma_constraint=None)(x_psd)
    x_psd = keras.layers.ReLU()(x_psd)

    x = keras.layers.Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=psd_filter_nums*2, padding='same',
                            use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_psd)
    x = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                        beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(x)

    x_psd = keras.layers.Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=psd_filter_nums*2, padding='same',
                                use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_psd)
    x = keras.layers.add([x_psd, x])
    x = keras.layers.MaxPooling2D(pool_size=(
        4, 4), strides=(2, 2), padding='valid')(x)

    x = keras.layers.ReLU()(x)
    eeg_2D = keras.layers.Dropout(0.5)(x)

    input_layer_eog = keras.layers.Input(
        name='input_layer_time_eog', shape=(2, times))
    input_layer_eog2 = keras.layers.Permute((2, 1))(input_layer_eog) 
    eog_1D = deepsleepnet(input_layer_eog2, Fs,
                          time_filters_nums, bn_mom)

    fusion_out = SE_Fusion(eog_1D, eeg_2D, reduction_ratio=reduction_ratio,
                activation=se_activation)

    flip_layer = GradientReversal(lambda_reversal)
    G_d_in = flip_layer(fusion_out)
    G_d_out = keras.layers.Dense(128, activation='relu',
                           kernel_regularizer=l2(0.1))(G_d_in)
    domain_out = keras.layers.Dense(units=num_domain,
                           activation='softmax',
                           name='Domain')(G_d_out)

    out = keras.layers.Dense(128, activation='relu',
                           kernel_regularizer=l2(0.1))(fusion_out)
    softmax_out = keras.layers.Dense(num_class, activation='softmax',
                           kernel_regularizer=l2(0.1), name='Label')(out)
    model = keras.models.Model(
        [input_layer, input_layer_eog], [softmax_out, domain_out])

    test_model = keras.models.Model(
        [input_layer, input_layer_eog], softmax_out)
    return model, test_model

def create_model_test():
    num_classes = 5
    num_domain = 10
    seq_len = 5
    width = 16
    height = 16
    model, test_model = getModel(num_classes, num_domain, seq_len, width, height, time_filters_nums=128,
                                            psd_filter_nums=16, reduction_ratio=2, times=11520, Fs=128, se_activation='sigmoid')
    model.summary()
