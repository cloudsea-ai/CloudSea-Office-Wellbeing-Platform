from __future__ import print_function
import requests
import time
import json
import cv2

addr = 'http://localhost:5000'
url = addr + '/posture_recognition'

# prepare headers for http request
content_type = 'image/jpeg'
# headers = {'content-type': content_type}

def post_image(img_file):
    """ post image and return the response """
    # file_name = open('file_name.txt', 'wb')
    # file_name.write('fra_recline.jpeg')
    # file_name.close()
    # img = open(img_file, 'rb').read()
    # response = requests.post(test_url, data=img, headers=headers)
    response = requests.post(url, files={'image': open(img_file, 'rb')})

    return response

response = post_image('./sample_images/fra_straight.jpeg')
print(json.loads(response.text))
# print(post_image('./sample_images/fra_recline.jpeg'))
# print(response.text)
# response = post_image('./sample_images/fra_recline.jpeg')
# print(json.loads(response.text))
# img = cv2.imread('lena.jpg')
# # encode image as jpeg
# _, img_encoded = cv2.imencode('.jpg', img)
# # send http request with image and receive response
# response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# # decode response””
