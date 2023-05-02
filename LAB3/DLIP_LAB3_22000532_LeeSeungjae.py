import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture('road_lanechange_student.mp4')

if not cap.isOpened():
    print("Can't write video !!! check setting")
    quit()

fps = cap.get(cv.CAP_PROP_FPS)
w = round(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('DLIP_Lab3_22000532_LeeSeungjae.avi', fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()

    if ret:
        time = cap.get(cv.CAP_PROP_POS_MSEC)
        roi = frame[400:620, 280:1000]
        src = roi
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        # cv.imshow('gray', gray)

        inroi = np.array([(0, 220), (350, 20), (370, 20), (720, 220)], np.int32)
        mask = np.zeros_like(gray)
        cv.fillPoly(mask, [inroi], 255)
        mask = cv.bitwise_and(gray, mask)
        # cv.imshow('mask', mask)

        blur = cv.GaussianBlur(mask, (5, 5), 0)
        # cv.imshow('blur', blur)

        ret, thres = cv.threshold(blur, 125, 255, cv.THRESH_BINARY)
        # cv.imshow('thres', thres)

        element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        morph = cv.morphologyEx(thres, cv.MORPH_CLOSE, element)
        # cv.imshow('morph', morph)

        canny = cv.Canny(morph, 50, 150, None, 3)
        # cv.imshow('canny', canny)

        cdstP = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

        linesP = cv.HoughLinesP(canny, 1, np.pi / 180, 50, None, 40, 25)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv.LINE_AA) 
        # cv.imshow('cdstP', cdstP)

        cdstP_gray = cv.cvtColor(cdstP, cv.COLOR_BGR2GRAY)

        dst = frame.copy()
        crop = dst[400:220 + 400, 280:720 + 280]
        cv.copyTo(cdstP, cdstP_gray, crop)
        cv.imshow('dst', dst)

        out.write(dst)

        k = cv.waitKey(10)
        if k == 27:
            break
    else:
        print("Cannot find a frame from video stream")
        break

cap.release()
out.release()
cv.destroyAllWindows()
