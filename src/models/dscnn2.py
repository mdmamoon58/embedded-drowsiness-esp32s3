"""Depthwise-Separable CNN (DSCNN2) model definition.

Exposed function:
    create_dscnn_model(input_shape, num_classes)
"""
import tensorflow as tf


def _ds_block(x, filters, kernel_size=(3,3), strides=(1,1)):
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def create_dscnn_model(input_shape, num_classes: int):
    """Build a depthwise-separable CNN.

    Args:
        input_shape: 3-tuple (time, features, channels)
        num_classes: number of output classes
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = _ds_block(inputs, 32)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = _ds_block(x, 64)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = _ds_block(x, 128)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="dscnn2")
    return model
