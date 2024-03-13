
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



base_model = keras.applications.Xception(weights='imagenet', include_top=False)

def extractor_model():
    base_model.trainable=True
    for layer in base_model.layers[:100]:
        layer.trainable=False

    inputs = keras.Input(shape=(100, 100, 3))
    #x = keras.applications.resnet.preprocess_input(inputs)
    x = base_model(inputs)
    x = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(7, activation='softmax')(x)


    feature_extractor = keras.Model(inputs=inputs, outputs=x)

    return feature_extractor