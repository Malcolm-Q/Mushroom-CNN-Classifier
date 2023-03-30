from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.config.run_functions_eagerly(True)
from PIL import Image
import logging
from ModelLoader import load_siamese
from ModelLoader import load_baseline
logging.getLogger('tensorflow').setLevel(logging.ERROR)



# tiny function to prep our inputs
def prepImage(img):
    # crop to square if needed
    img = Image.fromarray(img)
    width, height = img.size
    if width > height:
        left = (width - height) / 2
        right = left + height
        top = 0
        bottom = height
    elif width < height:
        top = (height - width) / 2
        bottom = top + width
        left = 0
        right = width
        img = img.crop((left, top, right, bottom))

    # size to 200x200
    width, height = img.size
    if width != 200:
        img = img.resize((200,200), Image.ANTIALIAS)
    return img



class RawFeats:
    def __init__(self, feats):
        self.feats = feats

    def fit(self, X, y=None):
        pass


    def transform(self, X, y=None):
        return X[self.feats]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
app = Flask(__name__)
api = Api(app)

siamese_model, class_to_species = load_siamese()

class Prediction(Resource):
    def post(self):
        print('received')
        payload = request.get_json()

        img1 = prepImage(np.array(payload['array1']))
        img2 = prepImage(np.array(payload['array2']))

        prediction = siamese_model.predict([img1,img2]).tolist()

        return class_to_species[str(np.argmax(prediction[0]))] + '\n Confidence: ' + str(prediction[0][np.argmax(prediction[0])])

model = load_baseline()

class EnvironmentScan(Resource):
    def post(self):
        print('received')
        payload = request.get_json()

        img = prepImage(np.array(payload['array1']))

        prediction = model.predict(img)

        top5_indices = (-prediction[0]).argsort()[:5]
        top5_probs = prediction[0][top5_indices]

        top5_species = [class_to_species[str(index)] for index in top5_indices]

        result = [(species, float(prob)) for species, prob in zip(top5_species, top5_probs)]
        
        return result


    
api.add_resource(Prediction, '/prediction')
api.add_resource(EnvironmentScan, '/predictEnv')

app.run(debug=True, host='0.0.0.0', port=8080, use_reloader=False)