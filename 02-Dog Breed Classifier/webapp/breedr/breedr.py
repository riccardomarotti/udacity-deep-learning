import os, uuid, tempfile, base64, urllib.parse
from .ai import Brain
from PIL import Image
from resizeimage import resizeimage

from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, jsonify


app = Flask(__name__)
app.config.from_object(__name__)

app.config.update(dict(
    WEIGHTS_FILE='test.h5'
))
app.config.from_envvar('FLASKR_SETTINGS', silent=True)

tempdir = tempfile.gettempdir()
brain = Brain()

@app.route('/')
def upload_image():
    return render_template('upload_file.html')

@app.route('/show_breed/<id>')
def show_breed(id):
    with open(os.path.join(tempdir, id), mode='rb') as file:
        image_data = urllib.parse.quote(base64.b64encode(file.read()))

    detected_breed = brain.find_breed(os.path.join(tempdir, id))


    return render_template('show_breed.html', image_data=image_data, detected_breed=detected_breed)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = str(uuid.uuid4())
    file.save(os.path.join(tempdir, filename))

    data = {"id": filename}
    return jsonify(data)