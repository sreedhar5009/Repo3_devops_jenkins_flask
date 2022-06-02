import numpy as np
from flask import Flask, request, render_template
import pickle 

app = Flask(__name__, template_folder = 'templates')
model = pickle.load(open('temp.pkl','rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/sreeg', methods=['GET'])
def sreeg():
    return 'sreedhar reddy maddirala - GET GET'

@app.route('/sreep', methods=['POST'])
def sreep():
    return 'sreedhar reddy maddirala - POST POST'

@app.route('/pred', methods=['POST'])
def predict():
    inp_features = [float(x) for x in request.form.values()]
    final_features = [np.array(inp_features)]
    print('inp_features', inp_features, '\n final_features', final_features)
    prediction1 = model.predict(final_features)
    output = round(prediction1[0], 2)
    print('\n prediction1', prediction1, '\n prediction', output)
    # pred = model.predict([[28.500000,2,0]])[0]
    # output = round(prediction[0], 2)
    if output < 0:
        return render_template('index.html', output = "Predicted Price is negative, values entered not reasonable")
    elif output >= 0:
        return render_template('index.html', output = 'Predicted Price of the house is: ${}'.format(output))

if __name__ == "__main__":
    # app.run(debug=True, port = 5050)
    app.run(debug=False)
    # app.run(host='0.0.0.0', port = 8080)