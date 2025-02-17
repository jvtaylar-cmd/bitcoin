import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('crypto_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    data = request.form['opening_price'] 
    input = float(data)

	# convert the data into numpy array and perform prediction
    prediction = model.predict([[np.array(input)]])
    
    
    output = np.round(prediction[0], 2)

    return render_template('index.html', prediction_text='Crypto Price should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=False)