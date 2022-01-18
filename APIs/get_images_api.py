from flask import Flask, request, Response
import jsonpickle

import os

import sys
sys.path.append('../')
from posture_image import recognize_posture

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/posture_recognition', methods=['POST'])
def posture():
    r = request
    file = r.files['image']
    filename = file.filename
    filepath = '../tmp/' + filename
    file.save(filepath)

    id = os.path.splitext(os.path.basename(filepath))[0].split('_', 1)[0]
    tic = os.path.splitext(os.path.basename(filepath))[0].split('_', 1)[1]

    recognize_posture(filepath)

    # build a response dict to send back to client
    response = {'message': 'File: {} ID: {} Time: {} analysed. Results file updated'.format(filename, id, tic)
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

# start flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)