from tensorflow.keras import layers, activations, regularizers, preprocessing
from tensorflow.keras.layers import Conv3D, Conv2D, Dense, Flatten, LeakyReLU, Conv2DTranspose
from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, BatchNormalization, Add, ZeroPadding2D
from tensorflow_addons.layers.normalizations import InstanceNormalization
from tensorflow.keras.models import Model


class Discriminator:
    def __init__(self, width=1920, height=1080, channels=3, hidden_layers=3):
        self.width = width
        self.height = height
        self.channels = channels
        self.residual_blocks = residual_block
        self.hidden_layers = hidden_layers

    def build_model(self):
        input_shape = (self.width, self.height, self.channels)
        input_layer = Input(shape=input_shape)

        pad = ZeroPadding2D(padding=(1, 1))(input_layer)

        x = Conv2D(filters=64, kernel_size=4, strides=2, padding="valid")(pad)
        x = LeakyReLU(alpha=0.2)(x)

        x = ZeroPadding2D(padding=(1, 1))(x)

        for i in range(1, self.hidden_layers + 1):
            x = Conv2D(filters=2 ** i * 64, kernel_size=4, strides=2, padding="valid")(x)
            x = InstanceNormalization(axis=1)(x)
            x = LeakyReLU(alpha=.2)(x)
            x = ZeroPadding2D(padding=(1, 1))(x)

        output = Conv2D(filters=1, kernel_size=4, strides=1, activation="sigmoid")(x)

        model = Model(inputs=[input_layer], outputs=[output])

        return model

    def summary(self):
        smaple = self.build_model()
        print(smaple.summary())

    def save_model(self):
        pass


class Generator:
    def __init__(self, width=1920, height=1080, channels=3, residual_blocks=3):
        self.width = width
        self.height = height
        self.channels = channels
        self.residual_blocks = residual_blocks

    def build_model_generator(self):
        input_shape = (self.width, self.height, self.channels)
        input_layer = Input(shape=input_shape)
        print(input_layer.shape)
        # Convolution block
        x = Conv2D(filters=32, kernel_size=7, strides=1, padding="same")(input_layer)
        x = InstanceNormalization(axis=1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
        x = InstanceNormalization(axis=1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
        x = InstanceNormalization(axis=1)(x)
        x = Activation("relu")(x)

        # Residual block
        for _ in range(self.residual_blocks):
            x = residual_block(x)

        # Upsampling block
        x = Conv2DTranspose(filters=64, strides=2, kernel_size=3, padding='same', use_bias=False)(x)
        x = InstanceNormalization(axis=1)(x)
        x = Activation("relu")(x)

        x = Conv2DTranspose(filters=32, strides=2, kernel_size=3, padding='same', use_bias=False)(x)
        x = InstanceNormalization(axis=1)(x)
        x = Activation("relu")(x)

        x = Conv2DTranspose(filters=3, strides=1, kernel_size=7, padding='same', use_bias=False)(x)
        output_layer = Activation('tanh')(x)

        model = Model(inputs=[input_layer], outputs=[output_layer])

        return model

    def summary(self):
        pass

    def save_model(self):
        pass


def residual_block(x):
    input_layer = x
    res = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(input_layer)
    res = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(res)
    res = Activation('relu')(res)
    res = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(res)
    res = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(res)
    return Add()([res, x])


def fake_model():
    input_shape = (960, 540, 3)
    input_layer = Input(shape=input_shape)
    res = Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(input_layer)
    res = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(res)
    res = Activation('relu')(res)
    x = Conv2DTranspose(filters=64, strides=2, kernel_size=3, padding='same', use_bias=False)(res)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(filters=3, strides=1, kernel_size=7, padding='same', use_bias=False)(x)
    output_layer = Activation('tanh')(x)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model
