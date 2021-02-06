from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'Titanic-prediction-ada-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        Pclass = int(request.form['Pclass'])
        Sex = int(request.form['Sex'])
        Age = int(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = float(request.form['Parch'])
        Fare = float(request.form['Fare'])
        Cabin = int(request.form['Cabin'])
        Embarked = int(request.form['Embarked'])

        data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Cabin,Embarked]])
        my_prediction = classifier.predict(data)

        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)