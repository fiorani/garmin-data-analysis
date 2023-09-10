import os.path
import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)
app.debug = True

@app.route("/", methods=["GET", "POST"])
def index():
    
    with open('datasets/hr_dataset.pkl', 'rb') as f:
        X_hr = pickle.load(f)
    with open('datasets/watt_dataset.pkl', 'rb') as f:
        X_watt = pickle.load(f)

    if request.method == "POST":
        card = request.form.get("card")

        if card == "hr":
            inputs = []
            for column_name, dtype in X_hr.dtypes.items():
                if dtype == "int64":
                    value = int(request.form[column_name])
                elif dtype == "float64":
                    value = float(request.form[column_name])
                inputs.append(value)
            with app.open_resource("models/hr_model.bin", "rb") as f:
                hr_model = pickle.load(f)
            response = hr_model.predict(np.array(inputs).reshape(1, -1))[0]
            return render_template("index.html", hr_pred=response, X_hr=X_hr, X_watt=X_watt)

        elif card == "watt":
            inputs = []
            for column_name, dtype in X_watt.dtypes.items():
                if dtype == "int64":
                    value = int(request.form[column_name])
                elif dtype == "float64":
                    value = float(request.form[column_name])
                inputs.append(value)
            with app.open_resource("models/watt_model.bin", "rb") as f:
                watt_model = pickle.load(f)
            response = watt_model.predict(np.array(inputs).reshape(1, -1))[0]
            return render_template("index.html", w_pred=response, X_hr=X_hr, X_watt=X_watt)

    return render_template("index.html", X_hr=X_hr, X_watt=X_watt)

if __name__ == '__main__':
  app.run()
