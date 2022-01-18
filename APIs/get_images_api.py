from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
# import posture_image
from posture_image import recognize_posture
from model import get_testing_model
from config_reader import config_reader
import time

import sys

import os

# Initialize the Flask application
app = Flask(__name__)

# model = load_trained_model()

# route http posts to this method
@app.route('/posture_recognition', methods=['POST'])
def posture():
    r = request
    # convert string of image data to uint8
    # nparr = np.fromstring(r.data, np.uint8)
    file = r.files['image']
    filename = file.filename
    filepath = '../tmp/' + filename
    file.save(filepath)
    # npimg = np.fromstring(file, np.uint8)
    # img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # print(file)
    # filename = file.filename
    # model = get_testing_model()
    # model.load_weights('../model/keras/model.h5')
    # params, model_params = config_reader()
    # recognize_posture(filepath, model, params, model_params)
    recognize_posture(filepath)


    # print(file.filename)
    # file_name = np.fromstring(r.data, np.uint8)
    # print(file_name)
    # file_name = r.headers['content-disposition']
    # file.save(file_name)
    # file.save(re.findall("filename=(.+)", file_name)[0])
    # file.save(filename)

    # decode image
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': 'File name:{} analysed. Results file updated'.format(filename)
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

    # sys.modules[__name__].__dict__.clear()
# start flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)