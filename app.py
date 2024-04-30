import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import json
from tensorflow.keras.models import Sequential, model_from_json

json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")

app = Flask(__name__)
#model = pickle.load(open('emotion_model.h5', 'rb'))
"""from tensorflow.keras.models import load_model
json_file = open('model\emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")"""
print("Loaded model from disk")
#model = json.load('model\emotion_model.json')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Emotion detected: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)