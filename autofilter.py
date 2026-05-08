# Program to automatically edit aviation photos
# by Logan Hanson
# May 2026
# Lots of commented out code from testing, took up extra time with file writes and clogged results dir

#import rawpy
import re
import os
import csv
import cv2
import sys
import math
import time
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

matplotlib.use("TkAgg")

filepath = input("Enter source file path -> ")
destpath = input("Enter destination file path -> ")
ratio = input("Enter Desired Aspect Ratio -> ")
#writeC = input("Write output CSV? (y/n) -> ")
writeC = 'n'
rat = re.search(r'(\w+)[\/:](\w+)', ratio)
num = len(os.listdir(filepath))
start = time.perf_counter()
old = start
ct = int(0)
maxT = 0
minT = sys.float_info.max
minName = str("")
maxName = str("")

if(type(rat) != None):
    aspectx = int(rat.group(1))
    aspecty = int(rat.group(2))
else:
    aspectx, aspecty = int(0), int(0)

data = [{} for _ in range(num * 4)]
dNum = 0
if(writeC == 'y'): 
    if(os.path.exists('data.csv')): os.remove('data.csv')

for file in Path(filepath).iterdir():
    img = cv2.imread(filepath + "/" + file.name)
    print("Opened file " + '\x1b[1;33;40m' + file.name + '\x1b[0m' + " at " + filepath)
    height, width = img.shape[:2]
    print("Height: " + str(height) + " Width: " + str(width))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_demo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_demo = img
    final = img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_edges = cv2.Canny(img_blurred, 100, 200)
    plane_cascade = cv2.CascadeClassifier('cascade.xml')
    found = plane_cascade.detectMultiScale(img_gray, minSize=(20, 20))
    largest = 0
    largestcoords = [0,0]
    largestdims = [0,0]

    for (x, y, w, h) in found:
        #cv2.rectangle(img_demo, (x, y), (x + w, y + h), (0, 255, 0), 5)
        if((w * h) > largest):
            largestcoords = [x,y]
            largestdims = [w,h]
            largest = w * h
    
    X, Y = largestcoords
    W, H = largestdims
    print('\x1b[1;37;40m' + "largest box: " + '\x1b[0m' + str(W) + "," + str(H) + " at " + str(X) + "," + str(Y))

    boxdims = largestdims
    boxcoords = largestcoords
    for (x, y, w, h) in found:
        if((w*h) >= ((largest / (width * height)) * largest)):
            #cv2.rectangle(img_demo, (x, y), (x + w, y + h), (255, 0, 255), 5)
            if(x < boxcoords[0]):
                boxdims[0] = boxdims[0] + (boxcoords[0] - x)
                boxcoords[0] = x
            else:
                if ((w + (x - boxcoords[0])) > boxdims[0]):
                    boxdims[0] = (x + w) - boxcoords[0]

            if(y < boxcoords[1]):
                boxdims[1] = boxdims[1] + (boxcoords[1] - y)
                boxcoords[1] = y
            else:
                if((h + (y - boxcoords[1])) > boxdims[1]):
                    boxdims[1] = (y + h) - boxcoords[1]


    x, y = boxcoords
    w, h = boxdims

    print('\x1b[1;37;40m' + "bounding box: " + '\x1b[0m' + str(w) + "," + str(h) + " at " + str(x) + "," + str(y))
    #cv2.rectangle(img_demo, (X, Y), (X + W, Y + H), (255, 255, 0), 5)
    cv2.rectangle(img_demo, (x, y), (x + w, y + h), (0, 0, 255), 5)

    if(x > (width - (x + w))):
        buffx = width - (x + w)
    else:
        buffx = x

    if(y > (height - (y + h))):
        buffy = height - (y + h)
    else:
        buffy = y
    
    aspect = w / h
    print("Initial Aspect Ratio: " + str(aspect))
    
    cv2.rectangle(img_demo, (x - buffx, y - buffy), (x + (2 * buffx) + w, y + (2 * buffy) + h), (0, 0, 0), 5)
    print("Original Buffers: " + str(buffx) + "," + str(buffy))

    aspect = ((2 * buffx) + w) / ((2 * buffy) + h)
    print("Original Aspect Ratio: " + str(aspect))

    if(aspect > aspectx / aspecty):
        correctx = int(-1 * ((((aspectx / 2) * (h + (2 * buffy))) / (aspecty)) - (w / 2) - buffx))
    else:
        correctx = 0

    if(aspect < aspectx / aspecty):
        correcty = int(-1 * ((((aspecty) * (w + (2 * buffx))) / (aspectx * 2)) - (h / 2) - buffy))
    else:
        correcty = 0

    buffx = buffx - correctx
    buffy = buffy - correcty
    print("Resized Buffers: " + str(buffx) + "," + str(buffy))
    aspect = ((2 * buffx) + w) / ((2 * buffy) + h)
    print("New Aspect Ratio: " + str(aspect))

    startY, endY, startX, endX = y-buffy, y+h+buffy, x-buffx, x+w+buffx
    crop = img_rgb[startY:endY, startX:endX]
    #crop = img[startY:endY, startX:endX]
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_demo, (startX, startY), (startX + w + (2 * buffx), startY + h + (2 * buffy)), (255, 0, 0), 5)

    subj = img_rgb[y:y+h, x:x+w]
    subj = cv2.cvtColor(subj, cv2.COLOR_RGB2BGR)
    
    filename = re.match(r'[A-Za-z0-9_-]+', file.name)

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    sharpscore = cv2.Laplacian(subj, cv2.CV_64F).var()
    brightness = np.mean(crop_gray)
    blu, grn, red = cv2.split(crop)
    blulevel = np.mean(blu)
    grnlevel = np.mean(grn)
    redlevel = np.mean(red)

    sFactor = ((brightness - sharpscore) / 300) * 2
    sFactor = max(sFactor,0)
    factor = sFactor
    center = 1 + (4 * factor)
    factor = -1 * factor

    print('\x1b[1;37;40m' + "Sharpness Score: " + '\x1b[0m' + str(sharpscore))
    print('\x1b[1;37;40m' + "Brightness Score: " + '\x1b[0m' + str(brightness))
    print('\x1b[1;31;40m' + "Red Level: " + '\x1b[0m' + str(redlevel) + '\x1b[1;32;40m' + " Green Level: " + '\x1b[0m' + str(grnlevel) + '\x1b[1;34;40m' + " Blue Level: " + '\x1b[0m' + str(blulevel))
    print("Sharpness factor: " + str(sFactor))

    if(brightness > 100):
        textcolor = (0,0,0)
    else:
        textcolor = (255,255,255)

    kernel1 = np.array([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
    
    kernel2 = np.array([[0,-1/4,0],
                        [-1/4,2,-1/4],
                        [0,-1/4,0]])
    
    kernel3 = np.array([[0,-2,0],
                        [-2,9,-2],
                        [0,-2,0]])

    kernel4 = np.array([[0, factor, 0],
                        [factor, center, factor],
                        [0, factor, 0]])

    # sharp1 = cv2.filter2D(crop, -1, kernel1)
    # S1 = cv2.filter2D(img, -1, kernel1)
    # sharp2 = cv2.filter2D(crop, -1, kernel2)
    # S2 = cv2.filter2D(img, -1, kernel2)
    # sharp3 = cv2.filter2D(crop, -1, kernel3)
    # S3 = cv2.filter2D(img, -1, kernel3)

    # sharp4 = cv2.filter2D(crop, -1, kernel4)
    # S4 = cv2.filter2D(img, -1, kernel4)

    final_hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(final_hsv)
    avsat_orig = np.average(sat)

    maxi = max(redlevel, grnlevel, blulevel)
    clip = 2.0
    tile = (1, 1)
    brt = abs(int((brightness - sharpscore) / 10))
    #brt = max(1, (maxi - brightness) / (maxi / 10))
    av = abs(((redlevel + grnlevel + blulevel) / 3) - brightness)
    #parameter = 1 + (1 / math.sqrt(av))
    #parameter = (255 - avsat) / 125

    final = cv2.filter2D(crop, -1, kernel4)
    final_lab = cv2.cvtColor(final, cv2.COLOR_BGR2LAB)
    f_l, a, b = cv2.split(final_lab)
    clahe = cv2.createCLAHE(clipLimit = clip, tileGridSize = tile)
    c_l = clahe.apply(f_l)
    lumin = cv2.merge((c_l, a, b))
    final = cv2.cvtColor(lumin, cv2.COLOR_LAB2BGR)
    final = cv2.addWeighted(final, 1.0, np.zeros(final.shape, final.dtype), 0, brt) #np.zeros(final.shape, final.dtype)
    #final = cv2.convertScaleAbs(final, alpha = 1.25, beta = -25)

    final_hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(final_hsv)
    avsat = np.average(sat)
    parameter = (max(avsat, avsat_orig) / min(avsat, avsat_orig)) * 1.3
    #parameter = 1.5
    sat = sat.astype(np.float32)
    sat = sat * parameter
    sat = np.clip(sat, 0, 255).astype(np.uint8)

    # TODO: Fix saturation stuff

    #sat = sat * 2
    #sat = np.clip(sat, 0, 255)
    final_hsv = cv2.merge([hue, sat, val])
    final = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # cv2.putText(crop, "Sharpness Score: " + str(sharpscore), (20,100), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(crop, "Brightness Score: " + str(brightness), (20,200), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(crop, "Red Level: " + str(redlevel), (20,300), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 170 + (textcolor[2] / 3)), 5, cv2.LINE_8, False)
    # cv2.putText(crop, "Green Level: " + str(grnlevel), (20,400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 170 + (textcolor[1] / 3), 0), 5, cv2.LINE_8, False)
    # cv2.putText(crop, "Blue Level: " + str(blulevel), (20,500), cv2.FONT_HERSHEY_PLAIN, 5, (170 + (textcolor[0] / 3), 0, 0), 5, cv2.LINE_8, False)
    # cv2.putText(crop, "Avg Saturation (original): " + str(avsat_orig), (20, 600), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(crop, "Avg Saturation (edited): " + str(avsat), (20, 700), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(crop, "Saturation Parameter: " + str(parameter), (20, 800), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)

    # origShp = (sharpscore)
    # origBrt = brightness
    # origRed = redlevel
    # origGrn = grnlevel
    # origBlu = blulevel

    # sharpscore = cv2.Laplacian(S1[y:y+h, x:x+w], cv2.CV_64F).var()
    # brightness = np.mean(cv2.cvtColor(sharp1, cv2.COLOR_BGR2GRAY))
    # blu, grn, red = cv2.split(sharp1)
    # blulevel = np.mean(blu)
    # grnlevel = np.mean(grn)
    # redlevel = np.mean(red)
    # cv2.putText(sharp1, "Sharpness Score: " + str(sharpscore), (20,100), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(sharp1, "Brightness Score: " + str(brightness), (20,200), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(sharp1, "Red Level: " + str(redlevel), (20,300), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 170 + (textcolor[2] / 3)), 5, cv2.LINE_8, False)
    # cv2.putText(sharp1, "Green Level: " + str(grnlevel), (20,400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 170 + (textcolor[1] / 3), 0), 5, cv2.LINE_8, False)
    # cv2.putText(sharp1, "Blue Level: " + str(blulevel), (20,500), cv2.FONT_HERSHEY_PLAIN, 5, (170 + (textcolor[0] / 3), 0, 0), 5, cv2.LINE_8, False)

    # shp1Shp = sharpscore
    # shp1Brt = brightness
    # shp1Red = redlevel
    # shp1Grn = grnlevel
    # shp1Blu = blulevel

    # sharpscore = cv2.Laplacian(S2[y:y+h, x:x+w], cv2.CV_64F).var()
    # brightness = np.mean(cv2.cvtColor(sharp2, cv2.COLOR_BGR2GRAY))
    # blu, grn, red = cv2.split(sharp2)
    # blulevel = np.mean(blu)
    # grnlevel = np.mean(grn)
    # redlevel = np.mean(red)
    # cv2.putText(sharp2, "Sharpness Score: " + str(sharpscore), (20,100), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(sharp2, "Brightness Score: " + str(brightness), (20,200), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(sharp2, "Red Level: " + str(redlevel), (20,300), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 170 + (textcolor[2] / 3)), 5, cv2.LINE_8, False)
    # cv2.putText(sharp2, "Green Level: " + str(grnlevel), (20,400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 170 + (textcolor[1] / 3), 0), 5, cv2.LINE_8, False)
    # cv2.putText(sharp2, "Blue Level: " + str(blulevel), (20,500), cv2.FONT_HERSHEY_PLAIN, 5, (170 + (textcolor[0] / 3), 0, 0), 5, cv2.LINE_8, False)

    # shp2Shp = sharpscore
    # shp2Brt = brightness
    # shp2Red = redlevel
    # shp2Grn = grnlevel
    # shp2Blu = blulevel

    # sharpscore = cv2.Laplacian(S3[y:y+h, x:x+w], cv2.CV_64F).var()
    # brightness = np.mean(cv2.cvtColor(sharp3, cv2.COLOR_BGR2GRAY))
    # blu, grn, red = cv2.split(sharp3)
    # blulevel = np.mean(blu)
    # grnlevel = np.mean(grn)
    # redlevel = np.mean(red)
    # cv2.putText(sharp3, "Sharpness Score: " + str(sharpscore), (20,100), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(sharp3, "Brightness Score: " + str(brightness), (20,200), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(sharp3, "Red Level: " + str(redlevel), (20,300), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 170 + (textcolor[2] / 3)), 5, cv2.LINE_8, False)
    # cv2.putText(sharp3, "Green Level: " + str(grnlevel), (20,400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 170 + (textcolor[1] / 3), 0), 5, cv2.LINE_8, False)
    # cv2.putText(sharp3, "Blue Level: " + str(blulevel), (20,500), cv2.FONT_HERSHEY_PLAIN, 5, (170 + (textcolor[0] / 3), 0, 0), 5, cv2.LINE_8, False)

    # shp3Shp = sharpscore
    # shp3Brt = brightness
    # shp3Red = redlevel
    # shp3Grn = grnlevel
    # shp3Blu = blulevel

    # sharpscore = cv2.Laplacian(S4[y:y+h, x:x+w], cv2.CV_64F).var()
    # brightness = np.mean(cv2.cvtColor(sharp4, cv2.COLOR_BGR2GRAY))
    # blu, grn, red = cv2.split(sharp4)
    # blulevel = np.mean(blu)
    # grnlevel = np.mean(grn)
    # redlevel = np.mean(red)
    # cv2.putText(sharp4, "Sharpness Score: " + str(sharpscore), (20,100), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(sharp4, "Brightness Score: " + str(brightness), (20,200), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)
    # cv2.putText(sharp4, "Red Level: " + str(redlevel), (20,300), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 170 + (textcolor[2] / 3)), 5, cv2.LINE_8, False)
    # cv2.putText(sharp4, "Green Level: " + str(grnlevel), (20,400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 170 + (textcolor[1] / 3), 0), 5, cv2.LINE_8, False)
    # cv2.putText(sharp4, "Blue Level: " + str(blulevel), (20,500), cv2.FONT_HERSHEY_PLAIN, 5, (170 + (textcolor[0] / 3), 0, 0), 5, cv2.LINE_8, False)
    # cv2.putText(sharp4, "Sharpness Factor " + str(-1 * factor), (20,600), cv2.FONT_HERSHEY_PLAIN, 5, textcolor, 5, cv2.LINE_8, False)

    # shp4Shp = sharpscore
    # shp4Brt = brightness
    # shp4Red = redlevel
    # shp4Grn = grnlevel
    # shp4Blu = blulevel

    # N = int(0)
    # shpsc = float(0)
    # shpf = float(0)
    # for N in range (0, 4):
    #     if(N == 0): 
    #         shpsc = shp1Shp
    #         shpf = 1
    #     elif(N == 1): 
    #         shpsc = shp2Shp
    #         shpf = 0.25
    #     elif(N == 2): 
    #         shpsc = shp3Shp
    #         shpf = 2
    #     elif(N == 3):
    #         shpsc = shp4Shp
    #         shpf = -1 * factor
    #     data[dNum] = {'filename': file.name,
    #                   'og sharpness': origShp,
    #                   'sharpness mod': shpf,
    #                   'new sharpness': shpsc}
    #     dNum = dNum + 1

    # ret1 = destpath + "/" + filename.group() + "_" + str(aspectx) + "_" + str(aspecty) + "_rect.jpg"
    # ret2 = destpath + "/" + filename.group() + "_" + str(aspectx) + "_" + str(aspecty) + "_crop.jpg"
    # ret3 = destpath + "/" + filename.group() + "_" + str(aspectx) + "_" + str(aspecty) + "_subj.jpg"

    # cv2.cvtColor(img_demo, cv2.COLOR_RGB2BGR)

    #plt.imshow(img_demo)
    #plt.savefig(ret1)
    # cv2.imwrite(ret1, img_demo)
    # print("Saved image at " + ret1)

    # cv2.imwrite(ret2, crop)
    # print("Saved image at " + ret2)

    # cv2.imwrite(ret3, subj)
    # print("Saved image at " + ret3)

    # ret1 = destpath + "/" + filename.group() + "_" + str(aspectx) + "_" + str(aspecty) + "_shp1.jpg"
    # ret2 = destpath + "/" + filename.group() + "_" + str(aspectx) + "_" + str(aspecty) + "_shp2.jpg"
    # ret3 = destpath + "/" + filename.group() + "_" + str(aspectx) + "_" + str(aspecty) + "_shp3.jpg"
    # ret4 = destpath + "/" + filename.group() + "_" + str(aspectx) + "_" + str(aspecty) + "_autoshp.jpg"
    ret5 = destpath + "/" + filename.group() + "_" + str(aspectx) + "_" + str(aspecty) + "_FINAL.jpg" 

    # cv2.imwrite(ret1, sharp1)
    # print("Saved image at " + ret1)

    # cv2.imwrite(ret2, sharp2)
    # print("Saved image at " + ret2)

    # cv2.imwrite(ret3, sharp3)
    # print("Saved image at " + ret3)

    # cv2.imwrite(ret4, sharp4)
    # print("Saved image at " + ret4)

    cv2.imwrite(ret5, final)
    print("Saved image at " + ret5)

    # ret6 = destpath + "/" + filename.group() + "_" + str(aspectx) + "_" + str(aspecty) + "_edge.jpg"
    # cv2.imwrite(ret6, img_edges)
    # print("Saved image at " + ret6)

    fileTime = time.perf_counter()
    ft = fileTime - old
    old = fileTime
    num = num - 1
    ct = ct + 1
    avg = (fileTime - start) / ct
    expected = avg * num
    if(ft < minT): 
        minT = ft
        minName = file.name
    if(ft > maxT): 
        maxT = ft
        maxName = file.name

    expHr = math.floor(expected / 3600)
    expMn = math.floor((expected - (expHr * 3600)) / 60)
    expSc = round((expected - (expHr * 3600) - (expMn * 60)))

    print(f"Time for file {file.name}: {ft:.4f}")
    print(f"Elapsed time: {(fileTime - start):.4f}")
    print(f"Expected time left: {expected:.4f}")
    print(f"Expected time left: {expHr:02d}:{expMn:02d}:{expSc:02d}")
    print(f"Files processed: {ct}")
    print(f"Files left: {num}")

    #cv2.imshow("Original", img_rgb)
    #cv2.imshow("Gray", img_gray)
    #cv2.imshow("Rectangles", img_demo)
    #cv2.imshow("Cropped", crop)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows

print(f"Fastest file: {minName} ({minT:.4f})")
print(f"Slowest file: {maxName} ({maxT:.4f})")

if(writeC == 'y'):
    with open('data.csv', 'w', newline='') as csvfile:
        fieldnames = ['filename',
                      'og sharpness',
                      'sharpness mod',
                      'new sharpness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
