import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


app = Flask(__name__, static_url_path='/static', template_folder='template')

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print(request.form.values())
    int_features = [float(x) for x in request.form.values()]

    finalInput = pd.DataFrame(data=int_features)
    finalInput = finalInput.T

    finalInput.rename(columns={0: 'Income composition of resources'}, inplace=True)
    finalInput.rename(columns={1: 'Country'}, inplace=True)
    finalInput.rename(columns={2: 'Adult Mortality'}, inplace=True)
    finalInput.rename(columns={3: 'BMI'}, inplace=True)
    finalInput.rename(columns={4: 'HIV/AIDS'}, inplace=True)
    finalInput.rename(columns={5: 'Schooling'}, inplace=True)

    # print(finalInput)

    # final_features = [np.array(int_features)]
    # print(final_features)
    prediction = model.predict(finalInput)
    
    output = round(prediction[0])
    # render the template and pass the data to the template
    # print(output)
    # return render_template('index.html', prediction_text='Life Expectancy should be {} years'.format(output))
    
    return render_template('index.html', prediction_text='{} '.format(output))


if __name__ == "__main__":
    app.run(debug=True)
