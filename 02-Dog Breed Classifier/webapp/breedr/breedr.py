import os, uuid, tempfile

from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, jsonify


app = Flask(__name__)
app.config.from_object(__name__)

app.config.update(dict(
    WEIGHTS_FILE='test.h5'
))
app.config.from_envvar('FLASKR_SETTINGS', silent=True)

tempdir = tempfile.gettempdir()


@app.route('/')
def upload_image():
    return render_template('upload_file.html')

@app.route('/show_breed/<id>')
def show_breed(id):
    return id

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = str(uuid.uuid4())
    file.save(os.path.join(tempdir, filename))
    data = {"id": filename}
    return jsonify(data)
