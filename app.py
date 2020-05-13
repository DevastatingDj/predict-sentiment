import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    sentence = int_features[0];
    score = analyser.polarity_scores(sentence)
    output = "{:-<30} {}".format(sentence, str(score))

    return render_template('index.html', prediction_text='The sentiment is calculate as $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)