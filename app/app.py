# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, flash
from config import *
from utils import *
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/s')
def home():
    return render_template('app_frontend.html', prediction_text="")

# define the logic for reading the inputs from the WEB PAGE, 
# running the model, and displaying the prediction
@app.route('/result', methods=['GET','POST'])
def predict():

    # get the description submitted on the web page
    a_description = request.form.get('description')
    prep, pred = test(a_description, path_dict, path_A, path_B, tag_count)
    return render_template('app_frontend.html',text=prep, prediction_text=pred)
    
if __name__ == '__main__':
    app.run()

