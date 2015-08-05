#! /usr/bin/env python
import os
import sys
import csv
import cv2
import glob
import numpy as np

if __name__ == "__main__":

    cv2.namedWindow('Lane Markers')
    imgs = glob.glob("images/*.png")
    
    intercepts = []
    
    for fname in imgs:
        # Load image and prepare output image
        img = cv2.imread(fname)
        
        # Draw sample lane markers
        (height, width) = img.shape[:2]
        left_x = width / 6
        right_x = width / 6 * 5
        color = (0,255,255) # yellow
        cv2.line(img, (left_x,height), (width/2-30, height/2), color)
        cv2.line(img, (right_x,height), (width/2+30, height/2), color)

        # Sample intercepts
        intercepts.append((os.path.basename(fname), left_x, right_x))

        # Show image
        cv2.imshow('Lane Markers', img)
        key = cv2.waitKey(50)
        if key == 27:
            sys.exit(0)
                
    # CSV output
    with open('intercepts.csv', 'w') as f:
        writer = csv.writer(f)    
        writer.writerows(intercepts)
        
    cv2.destroyAllWindows();
    	
