import flask
from flask import request, jsonify, abort
from werkzeug.utils import secure_filename
import os
import imageEnhancement
import convNetwork

app = flask.Flask(__name__)
app.config["DEBUG"] = True
UPLOAD_FOLDER = "./tmp"
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def home():
    return "<h1>Retinal Diseases Neural Net</h1><p>Detect and prevent retinal diseases with a smart neural network</p>"

# A route to return all of the available entries in our catalog.
@app.route('/api/v1/analyze/exGet', methods=['GET'])
def exGet():
    abort(404)

# A route to return all of the available entries in our catalog.
@app.route('/api/v1/analyze/retina', methods=['POST'])
def analyzeRetina():
    # check if the post request has the file part
    if 'file' not in request.files:
        return '', 400

    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return '', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        imageEnhancement.convertSingleImage(filename)
        prediction = convNetwork.predictImage("./tmp/" + filename)
        return jsonify({'status': 'success', 'prediction': prediction}), 200
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

@app.errorhandler(400)
def invalid_parameters(e):
    result = {'httpStatus': '400', 'name':e.name, 'description': e.description}
    return jsonify(result), 400

@app.errorhandler(500)
def internal_server_error(e):
    result = {'httpStatus': '500', 'name':e.name, 'description': e.description}
    return jsonify(result), 500

@app.errorhandler(Exception)
def defaultHandler(e):
    result = {'httpStatus': '500', 'name':e.name, 'description': e.description}
    return jsonify(result), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.run(host='192.168.0.40', port=8080)