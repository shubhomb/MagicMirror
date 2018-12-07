import numpy as np
import math
import copy
import cv2
import joblib
import os
from sklearn.svm import SVC

labels = {
    'left':0,
    'ok':1,
    'peace':2,
    'right':3,
    'spidey':4
}
index_key = list(zip(labels.values(),labels.keys()))

# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
global threshold
threshold = 60  # BINARY threshold
blurValue = 7  # GaussianBlur parameter
bgSubThreshold = 50
global bgModel
global bbox

global isBgCaptured
SAVEX = 128 # make sure this is same as cnn.py
SAVEY = 128

learningRate = 0
# variables
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))


def defineBackground():
    global bgModel
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    global isBgCaptured
    isBgCaptured = 1

def removeBG(frame):
    global bgModel
    defineBackground()
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# parameters
def defineBackground():
    global bgModel
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    global isBgCaptured
    isBgCaptured = 1


# Helper function for applying a mask to an array
def mask_array(array, imask):
    if array.shape[:2] != imask.shape:
        raise Exception("Shapes of input and imask are incompatible")
    output = np.zeros_like(array, dtype=np.uint8)
    for i, row in enumerate(imask):
        output[i, row] = array[i, row]
    return output


def cameraWork(model):
    global threshold
    global isBgCaptured
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open video")
        return None
    camera.set(10, 200)
    cv2.namedWindow('trackbar')
    pause = True

    while camera.isOpened():
        ret, frame = camera.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow('original', frame)
        # Get foreground from mask
        # Opening, closing and dilation
        # Get mask indexes

        drawing = None
        #  Main operation
        if isBgCaptured == 1:  # this part wont run until background captured
            img = removeBG(frame)
            foreground_display = copy.copy(img)


            img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

            # convert the image into binary image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('ori', thresh)
            drawing_r = cv2.resize(thresh, (SAVEX, SAVEY))
            gesture_recognize(model,drawing_r)

            # get the coutours
            thresh1 = copy.deepcopy(thresh)
            _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i

                res = contours[ci]
                hull = cv2.convexHull(res)
                drawing = np.zeros(img.shape, np.uint8)
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

                isFinishCal, cnt = calculateFingers(res, drawing)
                global triggerSwitch
                if triggerSwitch is True:
                    if isFinishCal is True and cnt <= 2:
                        print(cnt)
                        # app('System Events').keystroke(' ')  # simulate pressing blank space
            cv2.imshow('output', drawing)

        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit
            cv2.destroyAllWindows()
            break
        elif k == ord('b'):  # press 'b' to capture the background
            defineBackground()
            print('!!!Background Captured!!!')
        elif k == ord('r'):  # press 'r' to reset the background
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0

            print('!!!Reset BackGround!!!')
        elif k == ord('n'):
            triggerSwitch = True
        elif k == ord('w'):
            threshold = threshold + 5
            print ('!!!Threshold: %d' %threshold)
        elif k == ord('q'):
            threshold = threshold - 5
            print ('!!!Threshold: %d' %threshold)
        elif k == ord('p'):
            pause = not pause
            print (pause,' pause status')


    print('!!!Trigger On!!!')
    return camera

def gesture_recognize(model,img):
    img_r = np.reshape(img, [-1, SAVEX * SAVEY])
    out = model.predict(img_r)
    pred = index_key[out[0]]
    print (pred[1])
    print ('Predicted %s' %pred[1] )
    return out


def calculateFingers(res, drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)

        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                    cv2.circle(drawing, start, 8, [255, 0, 255], -1)

            return True, cnt
    return False, 0

if __name__ == '__main__':
    svm = joblib.load(os.path.join(os.path.split(os.getcwd())[0],'gesture_svm_linear.joblib'))
    print ('SVM loaded!')
    camera = cameraWork(svm)