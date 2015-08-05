'''
Created on Feb 15, 2015

@author: Tony
'''
import os
import sys
import csv
import cv2
import glob
import time
import numpy
import matplotlib.pyplot as pyplot

sTime = time.time()

with open('output' +os.sep+ 'Intercepts.txt') as interceptsFileObj:
    interceptsRead = csv.reader(interceptsFileObj, delimiter='\t')
    interceptsData = list(interceptsRead)

ImgList = glob.glob('images' +os.sep+ '*.png')
for i in xrange(0, len(ImgList)):
    
    if (ImgList[i] != interceptsData[i+1][0]):
        print 'Error: Mismatch between image and intercepts data!'
        sys.exit()
    else:
        img = numpy.average(cv2.imread(ImgList[i]), axis = 2)
        img = img[0:-1:1, 0:-1:1]
        
        intercepts = interceptsData[i+1][1:3]
        
        pyplot.imshow(img, cmap='gray', aspect='equal', interpolation=None)

        if (intercepts[0] == 'nan'): 
            pyplot.plot( 200, 1000, 'rs', markersize = 10 )
        else:
            pyplot.plot( intercepts[0], img.shape[0]-1, 'ys', markersize = 10 )
            
        if (intercepts[1] == 'nan'):
            pyplot.plot( 1400, 1000, 'rs', markersize = 10 )
        else:
            pyplot.plot( intercepts[1], img.shape[0]-1, 'ys', markersize = 10 )
            

        pyplot.ylim(0, 1210)
#         pyplot.xlim(-0, 1600)
        pyplot.gca().invert_yaxis()
        pyplot.show()



print("Duration: %s" % (time.time() - sTime))
    
    
    
    
    
