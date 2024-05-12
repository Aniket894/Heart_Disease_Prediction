from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Importing ridge regression model
Log_reg_model = pickle.load(open("modals/log.pkl", "rb"))

@app.route("/")
def hello_world():
    return "<h1>Heart Disease Classification Model</h1>"

@app.route("/predict_data", methods=['GET','POST'])
def index():
    if request.method=="POST":
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        chest_pain_type = int(request.form['chest_pain_type'])
        resting_bps = int(request.form['resting_bps'])
        cholesterol = int(request.form['cholesterol'])
        fasting_blood_sugar = int(request.form['fasting_blood_sugar'])
        resting_ecg = int(request.form['resting_ecg'])  # Fix typo here
        max_heart_rate = int(request.form['max_heart_rate'])
        exercise_angina = int(request.form['exercise_angina'])
        oldpeak = float(request.form['oldpeak'])
        ST_slope = int(request.form['ST_slope'])

        new_data = [[age, sex, chest_pain_type, resting_bps, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope]]

        try:
            result = Log_reg_model.predict(new_data)
            return render_template('result.html', result=result[0])
        except Exception as e:
            # Handle errors gracefully
            return render_template('error.html', error_message=str(e))

    else:
        return render_template("index.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")
