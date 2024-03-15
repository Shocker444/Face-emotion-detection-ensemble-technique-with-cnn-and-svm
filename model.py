
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# kernel_initializer
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode='fan_avg', distribution='uniform'
    )

class Attention(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        #self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        #height = tf.shape(inputs)[1]
        #width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        #inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        #attn_score = tf.einsum("bh, bH->bhH", q, k) * scale
        attn_score = tf.matmul(q, k, transpose_b=True) * scale
        #attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        #attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        #attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        #proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        context_vector = tf.matmul(attn_score, v)
        context_vector = self.proj(context_vector)
        #proj = self.proj(proj)
        return inputs + context_vector


#feature extractor
def extractor_model():
    base_model = keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-49]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = Attention(1024)(x)
    x = layers.Dense(7, activation='softmax')(x)

    feature_extractor = keras.Model(base_model.input, x)

    return feature_extractor