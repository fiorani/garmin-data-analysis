import os.path
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
app.debug = True

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    inputs = [
        float(request.args["LIMIT_BAL"]),
        int(request.args["PAY_6"]),
        int(request.args["EDUCATION"])
    ]
    with app.open_resource("models/model.bin", "rb") as f:
        model = pickle.load(f)
    response = model.predict([inputs])[0]
    return render_template("predict.html", resp=response)

if __name__ == '__main__':
  app.run()
