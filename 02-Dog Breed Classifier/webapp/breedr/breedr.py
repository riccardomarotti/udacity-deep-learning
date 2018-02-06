import os, uuid, tempfile, base64, urllib.parse
from .ai import Brain

from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, jsonify

from subprocess import call


app = Flask(__name__)
app.config.from_object(__name__)

app.config.update(dict(
    WEIGHTS_FILE='test.h5'
))
app.config.from_envvar('FLASKR_SETTINGS', silent=True)

tempdir = tempfile.gettempdir()
brain = Brain('saved_models/weights.best.xception.hdf5')

@app.route('/')
def upload_image():
    return render_template('upload_file.html')

@app.route('/show_breed/<id>')
def show_breed(id):
    full_local_filename = os.path.join(tempdir, id)

    if not os.path.isfile(full_local_filename):
        return render_template('image_not_present.html')

    with open(full_local_filename, mode='rb') as file:
        image_data = urllib.parse.quote(base64.b64encode(file.read()))

    message, breed = brain.find_breed(full_local_filename)
    query = duckduck_query_for(breed)

    os.remove(full_local_filename)

    return render_template('show_breed.html', image_data=image_data, message=message, query=query, breed=breed)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = str(uuid.uuid4())
    full_local_filename = os.path.join(tempdir, filename)
    file.save(full_local_filename)

    call(["mogrify", "-resize", "100000@", full_local_filename])



    data = {"id": filename}
    return jsonify(data)

def duckduck_query_for(breed):
    if breed:
        query = "+".join(breed.split())
        return "https://duckduckgo.com/?q={}&iax=images&ia=images".format(query)

    return None
