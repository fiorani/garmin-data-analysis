# %pip install numpy
# %pip install pandas
# %pip install matplotlib
# %pip install scikit-learn
# %pip install lightgbm
# %pip install xgboost
# %pip install flask
# %pip install xlrd
import os.path
import pickle
from flask import Flask, request, render_template


app = Flask(__name__)

notebook_dir = os.getcwd()
notebook = notebook_dir + os.sep + "garmin_analysis.ipynb"

@app.route('/')
def index():
  with open(notebook, 'r') as f:
    notebook_content = f.read()
  return render_template('index.html', notebook_content=notebook_content)

if __name__ == '__main__':
  app.run()