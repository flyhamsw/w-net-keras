import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import MeanIoU

"""
My implementation of W-Net (Hou et al., 2020)
Read "From W-Net to CDGAN: Bi-temporal Change Detection via Deep Learning Techniques" for details.
Please understand some configurations and hyper-parameters may be different from the paper.
====================================================================================================
Implemented by Sangwoo Ham, Ph.D student.
Lab. for Sensor and Modeling, Dept. of Geoinformatics,
University of Seoul, Korea
"""


def encoder_factory(input_channels, filters_1, filters_2, name):
    # input = keras.Input(shape=input_image_shape)
    input = keras.Input(shape=(None, None, input_channels))
    z = layers.Conv2D(filters_1, 3, 1, 'same', kernel_initializer='he_normal')(input)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)
    z = layers.Conv2D(filters_2, 3, 2, 'same', kernel_initializer='he_normal')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)
    return keras.Model(input, z, name=name)


def decoder_factory(input_channels, filters_1, filters_2, name, is_last=False):
    # input = keras.Input(shape=input_image_shape)
    input = keras.Input(shape=(None, None, input_channels))
    z = layers.Convolution2DTranspose(filters_1, 3, 1, 'same', kernel_initializer='he_normal')(input)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)
    z = layers.Convolution2DTranspose(filters_2, 3, 2, 'same', kernel_initializer='he_normal')(z)
    z = layers.BatchNormalization()(z)
    if is_last:
        z = layers.Activation('softmax')(z)
    else:
        z = layers.Activation('relu')(z)
    return keras.Model(input, z, name=name)


def build_w_net_siam(input_image_shape=(512, 512, 3), num_of_classes=2):
    encoder_1 = encoder_factory(3, 64, 128, name='encoder_1')
    encoder_2 = encoder_factory(128, 256, 512, name='encoder_2')
    encoder_3 = encoder_factory(512, 512, 512, name='encoder_3')
    encoder_4 = encoder_factory(512, 512, 512, name='encoder_4')

    decoder_1 = decoder_factory(512+512, 512, 512, name='decoder_1')
    decoder_2 = decoder_factory(512+512+512, 512, 512, name='decoder_2')
    decoder_3 = decoder_factory(512+512+512, 256, 128, name='decoder_3')
    decoder_4 = decoder_factory(128+128+128, 64, 2, name='decoder_4', is_last=True)

    input_1 = keras.Input(input_image_shape, name='input_1')
    input_2 = keras.Input(input_image_shape, name='input_2')

    encoded_img_1_conv_2 = encoder_1(input_1)
    encoded_img_1_conv_4 = encoder_2(encoded_img_1_conv_2)
    encoded_img_1_conv_6 = encoder_3(encoded_img_1_conv_4)
    encoded_img_1_conv_8 = encoder_4(encoded_img_1_conv_6)

    encoded_img_2_conv_2 = encoder_1(input_2)
    encoded_img_2_conv_4 = encoder_2(encoded_img_2_conv_2)
    encoded_img_2_conv_6 = encoder_3(encoded_img_2_conv_4)
    encoded_img_2_conv_8 = encoder_4(encoded_img_2_conv_6)

    concat_1 = layers.Concatenate()([encoded_img_1_conv_8, encoded_img_2_conv_8])

    deconv_2 = decoder_1(concat_1)
    deconv_2 = layers.Concatenate(axis=-1)([deconv_2, encoded_img_1_conv_6, encoded_img_2_conv_6])

    deconv_4 = decoder_2(deconv_2)
    deconv_4 = layers.Concatenate(axis=-1)([deconv_4, encoded_img_1_conv_4, encoded_img_2_conv_4])

    deconv_6 = decoder_3(deconv_4)
    deconv_6 = layers.Concatenate(axis=-1)([deconv_6, encoded_img_1_conv_2, encoded_img_2_conv_2])

    deconv_8 = decoder_4(deconv_6)

    # https://stackoverflow.com/questions/61824470/dimensions-mismatch-error-when-using-tf-metrics-meaniou-with-sparsecategorical
    class SparseMeanIoU(MeanIoU):
        def __init__(self,
                     y_true=None,
                     y_pred=None,
                     num_classes=None,
                     name=None,
                     dtype=None):
            super(SparseMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred = tf.math.argmax(y_pred, axis=-1)
            return super().update_state(y_true, y_pred, sample_weight)

    model = keras.Model(inputs=(input_1, input_2), outputs=deconv_8)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[SparseMeanIoU(num_classes=num_of_classes)]
    )

    return model


if __name__ == '__main__':
    my_model = build_w_net_siam()
    print(my_model.summary())
