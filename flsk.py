import os.path
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
app.debug = True

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        inputs = [
            float(request.form["distance[m]"]),
            float(request.form["altitude[m]"]),
        ]
        with app.open_resource("models/model.bin", "rb") as f:
            model = pickle.load(f)
        response = model.predict([inputs])[0]
        return render_template("index.html", resp=response)

if __name__ == '__main__':
  app.run()
