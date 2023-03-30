# For whatever reason creating the architecture and loading the weights is WAY faster so we'll be doing that.

import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.config.run_functions_eagerly(True)
from keras import layers
from tensorflow.image import resize
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

def load_siamese():
    img_augmentation = keras.models.Sequential(
        [
            layers.RandomRotation(factor=0.15),
            layers.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )


    inputs = layers.Input(shape=(200, 200, 3))
    x = resize(inputs, size=(260, 260))
    x = img_augmentation(x)
    model1 = keras.applications.efficientnet.EfficientNetB2(include_top=False, input_tensor=x, weights="imagenet")

    model1.trainable = False


    # New simple head, you can play around with the options here if you like but this does a decent job.
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model1.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.05
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Dense(32,name='dense1')(x)
    x = layers.LeakyReLU(alpha=(0.05),name='active1')(x)
    x = layers.Dense(64,name='dense2')(x)
    x = layers.LeakyReLU(alpha=(0.05),name='active2')(x)
    x = layers.Dense(128,name='dense3')(x)
    x = layers.LeakyReLU(alpha=(0.05),name='active3')(x)

    outputs = layers.Dense(30, activation="softmax", name="pred")(x)

    model1 = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    model1.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # load weights
    model1.load_weights('baseline_weights.h5')

    # Cut off the head
    model1 = keras.models.Model(model1.input, model1.layers[-2].output)

    # Clone the model
    model2 = keras.models.clone_model(model1)
    model2.set_weights(model1.get_weights())
    model2._name = 'model2'
    for layer in model2.layers:
        layer._name = layer.name + '_2'

    # Freeze the bodies
    model1.trainable=False
    model2.trainable=False

    # Hook up a new head
    concat = keras.layers.Concatenate()

    merged = concat([model1.output, model2.output])

    d1 = layers.Dense(32, name='denseHead1')(merged)
    lr1 = layers.LeakyReLU(alpha=(0.05), name='activeHead1')(d1)
    d2 = layers.Dense(64, name='denseHead2')(lr1)
    lr2 = layers.LeakyReLU(alpha=(0.05), name='activeHead2')(d2)
    head = keras.layers.Dense(30, activation='softmax',name='output')(lr2)

    siamese_model = keras.models.Model(inputs=[model1.input, model2.input], outputs=head)

    siamese_model.load_weights('siamese_weights.h5')

    class_to_species = {
            '0':'Amanita muscaria',
            '1':'Armillaria lutea',
            '2':'Auricularia auricula-judae',
            '3':'Bjerkandera adusta',
            '4':'Clitocybe nebularis',
            '5':'Coprinellus micaceus',
            '6':'Cuphophyllus virgineus',
            '7':'Daedaleopsis confragosa',
            '8':'Fomes fomentarius',
            '9':'Fomitopsis pinicola',
            '10':'Ganoderma applanatum',
            '11':'Gymnopilus penetrans',
            '12':'Hygrocybe miniata',
            '13':'Hypholoma fasciculare',
            '14':'Imleria badia',
            '15':'Leccinum scabrum',
            '16':'Lycoperdon perlatum',
            '17':'Meripilus giganteus',
            '18':'Mycena galericulata',
            '19':'Neoboletus luridiformis',
            '20':'Phaeolepiota aurea',
            '21':'Pleurotus ostreatus',
            '22':'Plicaturopsis crispa',
            '23':'Pluteus cervinus',
            '24':'Stereum hirsutum',
            '25':'Trametes gibbosa',
            '26':'Trametes hirsuta',
            '27':'Trametes versicolor',
            '28':'Tremella mesenterica',
            '29':'Tubaria furfuracea',
    }
    return siamese_model, class_to_species

def load_baseline():
    img_augmentation = keras.models.Sequential(
        [
            layers.RandomRotation(factor=0.15),
            layers.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )


    inputs = layers.Input(shape=(200, 200, 3))
    x = resize(inputs, size=(260, 260))
    x = img_augmentation(x)
    model1 = keras.applications.efficientnet.EfficientNetB2(include_top=False, input_tensor=x, weights="imagenet")

    model1.trainable = False


    # New simple head, you can play around with the options here if you like but this does a decent job.
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model1.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.05
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Dense(32,name='dense1')(x)
    x = layers.LeakyReLU(alpha=(0.05),name='active1')(x)
    x = layers.Dense(64,name='dense2')(x)
    x = layers.LeakyReLU(alpha=(0.05),name='active2')(x)
    x = layers.Dense(128,name='dense3')(x)
    x = layers.LeakyReLU(alpha=(0.05),name='active3')(x)

    outputs = layers.Dense(30, activation="softmax", name="pred")(x)

    model1 = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    model1.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # load weights
    model1.load_weights('baseline_weights.h5')

    return model1