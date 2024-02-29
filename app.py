from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('svm_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():
    age = request.form.get('age')
    Medu = request.form.get('Medu')
    studytime = request.form.get('studytime')
    failures = request.form.get('failures')
    goout = request.form.get('goout')
    health = request.form.get('health')
    absences = request.form.get('absences')
    G1 = request.form.get('G1')
    G2 = request.form.get('G2')
    sex_enc = request.form.get('sex_enc')
    higher_enc = request.form.get('higher_enc')

#result = {'age':age, 'Medu':Medu, 'studytime':studytime, 'failures':failures, 'goout':goout, 'health':health, 'absences': absences, 'G1':G1, 'G2':G2, 'sex_enc':sex_enc, 'higher_enc':higher_enc}
    input_query = np.array([[age,Medu,studytime,failures,goout,health,absences,G1,G2,sex_enc,higher_enc]])

    result = model.predict(input_query) [0]

    return jsonify(str(result))

if __name__ == '__main__':
    app.run(debug=True)