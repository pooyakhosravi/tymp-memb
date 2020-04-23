import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory

import uuid

from model_loader import ModelLoader

import os.path
import sys

import numpy as np



UPLOAD_FOLDER = 'uploads/'
MODEL_FOLDER = 'model/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

model_path = os.path.join(app.config['MODEL_FOLDER'], "tmpy_90")

model = ModelLoader(model_path)




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            extension = os.path.splitext(file.filename)[1]
            filename = secure_filename(str(uuid.uuid4()) +  extension)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filename)

            pred = model.predict(filename)

            pred_class = np.argmax(pred)
            pred_label = "Normal" if pred_class == 0 else "Abnormal"

            return '''
            <!doctype html>
            <h1>Model Predicitons<h1>

            <p>The model prediction is: {}<p>
            <br/>
            with probabilties of {}
            '''.format(pred_label, str(pred))

            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(debug=True, port=8000)