'''
    File name         : detectors.py
    Description       : Object detector used for detecting the objects in a video /image
    Python Version    : 3.7
'''

# Import python libraries
import numpy as np
import cv2


def detect(frame):
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Seuillage pour isoler le cercle noir
    _, img_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Trouver les contours
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    
    if contours:
        # Prendre le plus grand contour
        c = max(contours, key=cv2.contourArea)
        
        # Calculer le centre du contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return [cx, cy]

    return None