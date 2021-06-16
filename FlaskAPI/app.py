from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)
filename = 'finalized_model.sav'
# model=pickle.load(open('heart_atack_pre.pkl','rb'))
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict ():
    int_features=[int(x) for x in request.form.values()]
    # print(int_features)
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)[0]
    print(prediction)
    if prediction == 0:
        return render_template('home.html', prediction_text='This is a Legit transactions')
    else:
        return render_template('home.html', prediction_text='This is a Fruadulent transactions')



if __name__ =='__main__':
    app.run(debug=True)
