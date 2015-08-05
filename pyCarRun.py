'''
Created on Feb 13, 2015

@author: Tony
'''
import os
import cv2
import glob
import time
import numpy
from skimage import morphology
import matplotlib.pyplot as pyplot
from pyCar import pyLaneTracker
#===============================================================================

sTime = time.time()

ImgList = glob.glob('images' +os.sep+ '*.png')
# ImgList = glob.glob('tmp' +os.sep+ '*.png')
pyLaneTracker = pyLaneTracker('output')
Visualize = True
Write = False
pyLaneTracker.Intercepts = numpy.zeros( (len(ImgList), 2) )
pyLaneTracker.FileNames = []
InterceptsIdx = 0

for fname in ImgList:
    # Load image and average (all channels the same)...
    img = numpy.average(cv2.imread(fname), axis = 2)
    img = img[0:-1:1, 0:-1:1]
    mask = numpy.ones(img.shape)
    mask[0:numpy.round(img.shape[0]/2):1,:] = 0
    
    pyLaneTracker.FileNames.append(fname)
    pyLaneTracker.FileName = fname
    pyLaneTracker.Img = img * mask
    pyLaneTracker.denoiseImage()
    pyLaneTracker.Img = pyLaneTracker.Denoise
    pyLaneTracker.normalizeImage()
    pyLaneTracker.GaborOrientations = [100, 105, 110, 115, 120]
    pyLaneTracker.gaborFilter()
    pyLaneTracker.Img = pyLaneTracker.Gabor
    pyLaneTracker.normalizeImage()
    pyLaneTracker.thresholdImage()
    globalProcessed = pyLaneTracker.Threshold
    
    ### MORPHOLOGY ###
    globalProcessed = morphology.erosion(globalProcessed, morphology.disk(5))
    
    ### LINES  rho, theta, threshold, minLinLength, maxLineGap)  ###
    lines = numpy.zeros(3)
    lineCoords = cv2.HoughLinesP(globalProcessed.astype(numpy.uint8), 1, numpy.pi/180, 64, lines, minLineLength = 128, maxLineGap = 0)
    lineCoords = lineCoords[0,:,:]
    
    # x1,y1,x2,y2 in lines[0]:
    keepBool = numpy.zeros(lineCoords.shape[0]).astype(bool)
    lineLength = numpy.zeros(lineCoords.shape[0])
    lineSlope = numpy.zeros(lineCoords.shape[0])
    for i in xrange(0, lineCoords.shape[0]):
        Coords = lineCoords[i,:]
        lineLength[i] = numpy.sqrt( float(Coords[1] - Coords[3])**2 + float(Coords[0] - Coords[2])**2 )
        if (float(Coords[0] - Coords[2]) != 0):
            slope = float(Coords[1] - Coords[3]) / float(Coords[0] - Coords[2])
        else:
            slope = 0
        lineSlope[i] = slope
        if ((slope > 0.5) and (slope < 1.10)) or ((slope < -0.65) and (slope > -1.10)):
            keepBool[i] = True
            
    pyLaneTracker.Coordinates = lineCoords = lineCoords[keepBool,:]
    pyLaneTracker.Slopes = lineSlope = lineSlope[keepBool]
    lineLength = lineLength[keepBool]
    
    leftCorner = [0, globalProcessed.shape[0]]
    rightCorner = [globalProcessed.shape[1], globalProcessed.shape[0]]
    leftIntersects = []
    rightIntersects = []
    leftSlopes = []
    rightSlopes = []
    
    if Visualize:
        pyplot.subplot(1,3,1, aspect='equal'), pyplot.imshow(img, interpolation=None, cmap='gray') #pyplot.colorbar()
        pyplot.subplot(1,3,2, aspect='equal'), pyplot.imshow(numpy.log(pyLaneTracker.Gabor+1), interpolation=None, cmap='gray') #pyplot.colorbar()
        pyplot.subplot(1,3,3, aspect='equal'), pyplot.imshow(globalProcessed, interpolation=None, cmap='gray') #pyplot.colorbar()
    for i in xrange(0, lineCoords.shape[0]):
        
        B = lineCoords[i, 1] - (lineSlope[i] * lineCoords[i, 0])
        Xi = (pyLaneTracker.Img.shape[0] - B) / lineSlope[i]
        
        leftPoint = False
        rightPoint = False
        
        leftD = numpy.sqrt( (numpy.array([lineCoords[i,0], lineCoords[i,2]]) - leftCorner[0])**2 + (numpy.array([lineCoords[i,1], lineCoords[i,3]]) - leftCorner[1])**2 )
        rightD = numpy.sqrt( (numpy.array([lineCoords[i,0], lineCoords[i,2]]) - rightCorner[0])**2 + (numpy.array([lineCoords[i,1], lineCoords[i,3]]) - rightCorner[1])**2 )
        
        if any(leftD < 512):
            leftPoint = True
            leftIntersects.append(Xi)
            leftSlopes.append(lineSlope[i])
        elif any(rightD < 512):
            rightPoint = True
            rightIntersects.append(Xi)
            rightSlopes.append(lineSlope[i])
            
        if Visualize:
            if (lineLength[i] < 200) and (leftPoint or rightPoint):
                pyplot.subplot(1,3,3, aspect='equal'), pyplot.plot((Xi, lineCoords[i, 0]), (pyLaneTracker.Img.shape[0], lineCoords[i, 1]), 'ys-')
                pyplot.subplot(1,3,3, aspect='equal'), pyplot.plot((lineCoords[i,0], lineCoords[i,2]),(lineCoords[i,1], lineCoords[i,3]), 'r-')
            elif (leftPoint or rightPoint):
                pyplot.subplot(1,3,3, aspect='equal'), pyplot.plot((Xi, lineCoords[i, 0]), (pyLaneTracker.Img.shape[0], lineCoords[i, 1]), 'ys-')
                pyplot.subplot(1,3,3, aspect='equal'), pyplot.plot((lineCoords[i,0], lineCoords[i,2]),(lineCoords[i,1], lineCoords[i,3]), 'g-')
        
    if Visualize:
    #     pyplot.ylim(0, 1200)
    #     pyplot.xlim(0, 1600)
    #     pyplot.gca().invert_yaxis()
        pyplot.show()
    
    leftIntersects = numpy.asarray(leftIntersects)
    rightIntersects = numpy.asarray(rightIntersects)

    ###--- LEFT ---###
    if not any(leftIntersects):
        pyLaneTracker.Intercepts[InterceptsIdx, 0] = None
        print 'None'
    else:
        pyLaneTracker.Intercepts[InterceptsIdx, 0] = numpy.mean(leftIntersects)
        print 'Intercepts mean (left): '+ str(numpy.mean(leftIntersects))+ ' STD: '+ str(numpy.std(leftIntersects))
        
    ###--- RIGHT ---###
    if not any(rightIntersects):
        pyLaneTracker.Intercepts[InterceptsIdx, 1] = None
        print 'None'
    else:
        pyLaneTracker.Intercepts[InterceptsIdx, 1] = numpy.mean(rightIntersects)
        print 'Intercepts mean (right): '+ str(numpy.mean(rightIntersects))+ ' STD: '+ str(numpy.std(rightIntersects))
        
        
    InterceptsIdx += 1

if Write:
    pyLaneTracker.writeIntersections()
    
    
print("Duration: %s" % (time.time() - sTime))





