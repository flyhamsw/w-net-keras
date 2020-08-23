from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import he_normal

"""
My implementation of W-Net (Hou et al., 2020)
Please read "From W-Net to CDGAN: Bi-temporal Change Detection via Deep Learning Techniques" for details.

Implemented by Sangwoo Ham
Lab. for Sensor and Modeling, University of Seoul, Korea
"""

# Don't literally use these variables!
# Set input_image_shape, num_of_classes and bn considering your research goal.
input_image_shape = (224, 224, 3)
num_of_classes = 2
bn = True

input_1 = keras.Input(shape=input_image_shape)
input_2 = keras.Input(shape=input_image_shape)

my_initializer = he_normal()

conv_1_1 = layers.Convolution2D(64, 3, padding='same', kernel_initializer=my_initializer)(input_1)
conv_1_1 = layers.BatchNormalization()(conv_1_1) if bn else conv_1_1
conv_1_1 = layers.Activation('relu')(conv_1_1)

conv_1_2 = layers.Convolution2D(128, 3, strides=2, padding='same', kernel_initializer=my_initializer)(conv_1_1)
conv_1_2 = layers.BatchNormalization()(conv_1_2) if bn else conv_1_2
conv_1_2 = layers.Activation('relu')(conv_1_2)

conv_1_3 = layers.Convolution2D(256, 3, padding='same', kernel_initializer=my_initializer)(conv_1_2)
conv_1_3 = layers.BatchNormalization()(conv_1_3) if bn else conv_1_3
conv_1_3 = layers.Activation('relu')(conv_1_3)

conv_1_4 = layers.Convolution2D(512, 3, strides=2, padding='same', kernel_initializer=my_initializer)(conv_1_3)
conv_1_4 = layers.BatchNormalization()(conv_1_4) if bn else conv_1_4
conv_1_4 = layers.Activation('relu')(conv_1_4)

conv_1_5 = layers.Convolution2D(512, 3, padding='same', kernel_initializer=my_initializer)(conv_1_4)
conv_1_5 = layers.BatchNormalization()(conv_1_5) if bn else conv_1_5
conv_1_5 = layers.Activation('relu')(conv_1_5)

conv_1_6 = layers.Convolution2D(512, 3, strides=2, padding='same', kernel_initializer=my_initializer)(conv_1_5)
conv_1_6 = layers.BatchNormalization()(conv_1_6) if bn else conv_1_6
conv_1_6 = layers.Activation('relu')(conv_1_6)

conv_1_7 = layers.Convolution2D(512, 3, padding='same', kernel_initializer=my_initializer)(conv_1_6)
conv_1_7 = layers.BatchNormalization()(conv_1_7) if bn else conv_1_7
conv_1_7 = layers.Activation('relu')(conv_1_7)

conv_1_8 = layers.Convolution2D(512, 3, strides=2, padding='same', kernel_initializer=my_initializer)(conv_1_7)
conv_1_8 = layers.BatchNormalization()(conv_1_8) if bn else conv_1_8
conv_1_8 = layers.Activation('relu')(conv_1_8)

conv_2_1 = layers.Convolution2D(64, 3, padding='same', kernel_initializer=my_initializer)(input_2)
conv_2_1 = layers.BatchNormalization()(conv_2_1) if bn else conv_2_1
conv_2_1 = layers.Activation('relu')(conv_2_1)

conv_2_2 = layers.Convolution2D(128, 3, strides=2, padding='same', kernel_initializer=my_initializer)(conv_2_1)
conv_2_2 = layers.BatchNormalization()(conv_2_2) if bn else conv_2_2
conv_2_2 = layers.Activation('relu')(conv_2_2)

conv_2_3 = layers.Convolution2D(256, 3, padding='same', kernel_initializer=my_initializer)(conv_2_2)
conv_2_3 = layers.BatchNormalization()(conv_2_3) if bn else conv_2_3
conv_2_3 = layers.Activation('relu')(conv_2_3)

conv_2_4 = layers.Convolution2D(512, 3, strides=2, padding='same', kernel_initializer=my_initializer)(conv_2_3)
conv_2_4 = layers.BatchNormalization()(conv_2_4) if bn else conv_2_4
conv_2_4 = layers.Activation('relu')(conv_2_4)

conv_2_5 = layers.Convolution2D(512, 3, padding='same', kernel_initializer=my_initializer)(conv_2_4)
conv_2_5 = layers.BatchNormalization()(conv_2_5) if bn else conv_2_5
conv_2_5 = layers.Activation('relu')(conv_2_5)

conv_2_6 = layers.Convolution2D(512, 3, strides=2, padding='same', kernel_initializer=my_initializer)(conv_2_5)
conv_2_6 = layers.BatchNormalization()(conv_2_6) if bn else conv_2_6
conv_2_6 = layers.Activation('relu')(conv_2_6)

conv_2_7 = layers.Convolution2D(512, 3, padding='same', kernel_initializer=my_initializer)(conv_2_6)
conv_2_7 = layers.BatchNormalization()(conv_2_7) if bn else conv_2_7
conv_2_7 = layers.Activation('relu')(conv_2_7)

conv_2_8 = layers.Convolution2D(512, 3, strides=2, padding='same', kernel_initializer=my_initializer)(conv_2_7)
conv_2_8 = layers.BatchNormalization()(conv_2_8) if bn else conv_2_8
conv_2_8 = layers.Activation('relu')(conv_2_8)

concat_1 = layers.Concatenate(axis=-1)([conv_1_8, conv_2_8])

deconv_1 = layers.Convolution2DTranspose(512, 3, padding='same', kernel_initializer=my_initializer)(concat_1)
deconv_1 = layers.BatchNormalization()(deconv_1) if bn else deconv_1
deconv_1 = layers.Activation('relu')(deconv_1)

deconv_2 = layers.Convolution2DTranspose(512, 3, strides=2, padding='same', kernel_initializer=my_initializer)(deconv_1)
deconv_2 = layers.BatchNormalization()(deconv_2) if bn else deconv_2
deconv_2 = layers.Activation('relu')(deconv_2)

deconv_2 = layers.Concatenate(axis=-1)([deconv_2, conv_1_6, conv_2_6])

deconv_3 = layers.Convolution2DTranspose(512, 3, padding='same', kernel_initializer=my_initializer)(deconv_2)
deconv_3 = layers.BatchNormalization()(deconv_3) if bn else deconv_3
deconv_3 = layers.Activation('relu')(deconv_3)

deconv_4 = layers.Convolution2DTranspose(512, 3, strides=2, padding='same', kernel_initializer=my_initializer)(deconv_3)
deconv_4 = layers.BatchNormalization()(deconv_4) if bn else deconv_4
deconv_4 = layers.Activation('relu')(deconv_4)

deconv_4 = layers.Concatenate(axis=-1)([deconv_4, conv_1_4, conv_2_4])

deconv_5 = layers.Convolution2DTranspose(256, 3, padding='same', kernel_initializer=my_initializer)(deconv_4)
deconv_5 = layers.BatchNormalization()(deconv_5) if bn else deconv_5
deconv_5 = layers.Activation('relu')(deconv_5)

deconv_6 = layers.Convolution2DTranspose(128, 3, strides=2, padding='same', kernel_initializer=my_initializer)(deconv_5)
deconv_6 = layers.BatchNormalization()(deconv_6) if bn else deconv_6
deconv_6 = layers.Activation('relu')(deconv_6)

deconv_6 = layers.Concatenate(axis=-1)([deconv_6, conv_1_2, conv_2_2])

deconv_7 = layers.Convolution2DTranspose(num_of_classes, 3, strides=2, padding='same', kernel_initializer=my_initializer)(deconv_6)
deconv_7 = layers.BatchNormalization()(deconv_7) if bn else deconv_7
deconv_7 = layers.Activation('softmax')(deconv_7)

model = keras.Model(inputs=(input_1, input_2), outputs=deconv_7)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
)

print(model.summary())
