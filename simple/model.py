import tensorflow as tf

def Net(input_shape):
    _input = tf.keras.Input(shape=input_shape)
    net = _input
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(5, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    net = tf.keras.layers.Dense(15, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)
    _output = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.he_normal() )(net)

    model = tf.keras.Model(inputs=_input, outputs=_output)
    return model
