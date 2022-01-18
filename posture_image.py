import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
import keras

import os.path
import csv
import sys

tic=0
# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
#
# model = get_testing_model()
# model.load_weights('../model/keras/model.h5')


def process (input_image, params, model_params, model):
    # ''' Start of finding the Key points of full body using Open Pose.'''
    oriImg = cv2.imread(input_image)  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    for m in range(1):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])
        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
        output_blobs = model.predict(input_img)
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = [] #To store all the key points which a re detected.
    peak_counter = 0

    prinfTick(1) #prints time required till now.

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    prinfTick(2) #prints time required till now.
    print()
    position = checkPosition(all_peaks) #check position of spine.
    left_kneeling, right_kneeling = checkKneeling(all_peaks) #check whether kneeling oernot
    folding_hands = checkHandFold(all_peaks) #check whether hands are folding or not.
    canvas1 = draw(input_image,all_peaks) #show the image.
    return canvas1 , position, left_kneeling, right_kneeling, folding_hands


def draw(input_image, all_peaks):
    canvas = cv2.imread(input_image)  # B,G,R order
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    return canvas


def checkPosition(all_peaks):
    try:
        f = 0
        if (all_peaks[16]):
            a = all_peaks[16][0][0:2] #Right Ear
            f = 1
        else:
            a = all_peaks[17][0][0:2] #Left Ear
        b = all_peaks[11][0][0:2] # Hip
        angle = calcAngle(a,b)
        degrees = round(math.degrees(angle))
        if (f):
            degrees = 180 - degrees
        if (degrees<70):
            return 1
        elif (degrees > 110):
            return -1
        else:
            return 0
    except Exception as e:
        print("person not in lateral view and unable to detect ears or hip")


#calculate angle between two points with respect to x-axis (horizontal axis)
def calcAngle(a, b):
    try:
        ax, ay = a
        bx, by = b
        if (ax == bx):
            return 1.570796
        return math.atan2(by-ay, bx-ax)
    except Exception as e:
        print("unable to calculate angle")


def checkHandFold(all_peaks):
    try:
        if (all_peaks[3][0][0:2]):
            try:
                if (all_peaks[4][0][0:2]):
                    distance  = calcDistance(all_peaks[3][0][0:2],all_peaks[4][0][0:2]) #distance between right arm-joint and right palm.
                    armdist = calcDistance(all_peaks[2][0][0:2], all_peaks[3][0][0:2]) #distance between left arm-joint and left palm.
                    if (distance < (armdist + 100) and distance > (armdist - 100) ): #this value 100 is arbitary. this shall be replaced with a calculation which can adjust to different sizes of people.
                        print("Not Folding Hands")
                        folding_hands = 0
                    else:
                        print("Folding Hands")
                        folding_hands = 1
            except Exception as e:
                print("Folding Hands")
                folding_hands = 1
    except Exception as e:
        try:
            if(all_peaks[7][0][0:2]):
                distance  = calcDistance( all_peaks[6][0][0:2] ,all_peaks[7][0][0:2])
                armdist = calcDistance(all_peaks[6][0][0:2], all_peaks[5][0][0:2])
                # print(distance)
                if (distance < (armdist + 100) and distance > (armdist - 100)):
                    print("Not Folding Hands")
                    folding_hands = 0
                else:
                    print("Folding Hands")
                    folding_hands = 1
        except Exception as e:
            print("Unable to detect arm joints")
            folding_hands = 0

    return folding_hands

def calcDistance(a,b): #calculate distance between two points.
    try:
        x1, y1 = a
        x2, y2 = b
        return math.hypot(x2 - x1, y2 - y1)
    except Exception as e:
        print("unable to calculate distance")

def checkKneeling(all_peaks):
    left_kneeling = []
    right_kneeling = []
    f = 0
    if (all_peaks[16]):
        f = 1
    try:
        if(all_peaks[10][0][0:2] and all_peaks[13][0][0:2]): # if both legs are detected
            rightankle = all_peaks[10][0][0:2]
            leftankle = all_peaks[13][0][0:2]
            hip = all_peaks[11][0][0:2]
            leftangle = calcAngle(hip,leftankle)
            leftdegrees = round(math.degrees(leftangle))
            rightangle = calcAngle(hip,rightankle)
            rightdegrees = round(math.degrees(rightangle))
        if (f == 0):
            leftdegrees = 180 - leftdegrees
            rightdegrees = 180 - rightdegrees
        if (leftdegrees > 60  and rightdegrees > 60): # 60 degrees is trail and error value here. We can tweak this accordingly and results will vary.
            print ("Both Legs are in Kneeling")
            right_kneeling = 1
            left_kneeling = 1
        elif (rightdegrees > 60):
            print ("Right leg is kneeling")
            right_kneeling = 1
        elif (leftdegrees > 60):
            print ("Left leg is kneeling")
            left_kneeling = 1
        else:
            print ("Not kneeling")
            right_kneeling = 0
            left_kneeling = 0

    except IndexError as e:
        try:
            if (f):
                a = all_peaks[10][0][0:2] # if only one leg (right leg) is detected
            else:
                a = all_peaks[13][0][0:2] # if only one leg (left leg) is detected
            b = all_peaks[11][0][0:2] #location of hip
            angle = calcAngle(b,a)
            degrees = round(math.degrees(angle))
            if (f == 0):
                degrees = 180 - degrees
            if (degrees > 60):
                print ("Both Legs Kneeling")
                right_kneeling = 1
                left_kneeling = 1
            else:
                print("Not Kneeling")
                right_kneeling = 0
                left_kneeling = 0
        except Exception as e:
            print("legs not detected")
            right_kneeling = 0
            left_kneeling = 0

    return left_kneeling, right_kneeling



def showimage(img): #sometimes opencv will oversize the image when using using `cv2.imshow()`. This function solves that issue.
    #screen_res = 1280, 720 #my screen resolution.
    #scale_width = screen_res[0] / img.shape[1]
    #scale_height = screen_res[1] / img.shape[0]
    #scale = min(scale_width, scale_height)
    #window_width = int(img.shape[1] * scale)
    #window_height = int(img.shape[0] * scale)
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image', window_width, window_height)
    #cv2.imshow('image', img)
    cv2.imwrite('image.png', img)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

def prinfTick(i): #Time calculation to keep a trackm of progress
    toc = time.time()
    print ('processing time%d is %.5f' % (i,toc - tic))

def save_results(path, result):
    # header = 'Time, ID, straight, reclined, hunchback, left_kneeling, right_kneeling, folding_hands'
    # with open("test_results.csv", "a") as f:
    # 	f.write(str(results)+'\n')

    filename = path
    file_exists = os.path.isfile(filename)

    with open (filename, 'a') as csvfile:
        headers = ['TimeStamp', 'ID', 'Back_Straight', 'Back_Reclined', 'Back_Hunchback', 'Left_kneeling', 'Right_kneeling', 'Folding_hands']
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow({'TimeStamp': result[0], 'ID': result[1],
                         'Back_Straight': result[2], 'Back_Reclined': result[3], 'Back_Hunchback': result[4],
                         'Left_kneeling': result[5], 'Right_kneeling': result[6], 'Folding_hands': result[7]})

# def load_trained_model():
#
#     print('loading model...')
#     model = get_testing_model()
#     print('loading weights...')
#     model.load_weights('../model/keras/model.h5')
#
#     return model

def recognize_posture(filepath):
    id = os.path.splitext(os.path.basename(filepath))[0].split('_', 1)[0]
    tic = os.path.splitext(os.path.basename(filepath))[0].split('_', 1)[1]

    print('start processing...')
    # model = model_loaded
    # params, model_params = params, model_params
    # model = 0
    print('loading model...')
    model = get_testing_model()
    print('loading weights...')
    model.load_weights('../model/keras/model.h5')
    csv_path = '../results/results_csv_api.csv'

    print('analysing...')

    vi = False
    if (vi == False):
        time.sleep(2)
        params, model_params = config_reader()
        _, position, left_kneeling, right_kneeling, folding_hands = process(filepath, params, model_params, model)
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
    if hunchback == 0 and reclined == 0 and straight == 0:
        os.remove(filepath)
    else:
        result = [tic, id, straight, reclined, hunchback, left_kneeling, right_kneeling, folding_hands]
        save_results(csv_path, result)
        os.remove(filepath)

    keras.backend.clear_session()


# if __name__ == '__main__': #main function of the program
# 	tic = time.time()
# 	path_to_image = './photos/'
# 	print('start processing...')
#
# 	model = get_testing_model()
# 	model.load_weights('./model/keras/model.h5')
#
# 	vi=False
# 	if(vi == False):
# 	    time.sleep(2)
# 	    params, model_params = config_reader()
# 	    canvas, position, left_kneeling, right_kneeling, folding_hands = process(path_to_image + 'fra_hunchback.jpeg', params, model_params)
# 	    showimage(canvas)
# 		if (position == 1):
# 			print("Hunchback")
# 			hunchback=1
# 			reclined=0
# 			straight=0
# 		elif (position == -1):
# 			print("Reclined")
# 			hunchback=0
# 			reclined=1
# 			straight=0
# 		elif (position == 0):
# 			print("Straight")
# 			hunchback=0
# 			reclined=0
# 			straight=1
# 			# back = 0
# 		else:
# 			hunchback = 0
# 			reclined = 0
# 			straight = 0
# 	if hunchback == 0 and reclined == 0 and straight == 0:
# 		os.remove(path_to_image + '/fra_hunchback.jpeg')
# 	else:
# 		result = [tic, 'id', straight, reclined, hunchback, left_kneeling, right_kneeling, folding_hands]
# 		save_results(result)
# 		os.remove(path_to_image + '/fra_hunchback.jpeg')