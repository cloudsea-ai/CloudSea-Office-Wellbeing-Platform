from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
from posture_image import process, save_results
from model import get_testing_model
from config_reader import config_reader
import time

import os

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api_test_image', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    # nparr = np.fromstring(r.data, np.uint8)
    file = r.files['image']
    filename = file.filename

    tic = time.time()

    print('start processing...')

    model = get_testing_model()
    model.load_weights('./model/keras/model.h5')

    vi = False
    if (vi == False):
        time.sleep(2)
        params, model_params = config_reader()
        canvas, position, left_kneeling, right_kneeling, folding_hands = process(file, params, model_params)
        # showimage(canvas)
        if (position == 1):
            print("Hunchback")
            hunchback = 1
            reclined = 0
            straight = 0
        elif (position == -1):
            print("Reclined")
            hunchback = 0
            reclined = 1
            straight = 0
        elif (position == 0):
            print("Straight")
            hunchback = 0
            reclined = 0
            straight = 1
        # back = 0
        else:
            hunchback = 0
            reclined = 0
            straight = 0
    if hunchback != 0 or reclined != 0 or straight != 0:
        result = [tic, 'id', straight, reclined, hunchback, left_kneeling, right_kneeling, folding_hands]
        save_results(result)
        # os.remove(path_to_image + '/fra_hunchback.jpeg')

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
    # response = {'message': 'File name:{} received. size={}x{}'.format(file_name, img.shape[1], img.shape[0])
    #             }
    # encode response using jsonpickle
    # response_pickled = jsonpickle.encode(response)

    # return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)