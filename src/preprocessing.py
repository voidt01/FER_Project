import cv2

def preprocess(img, resize=(64, 64)):
    """
    return a processed image in Gray format
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize_img = cv2.resize(gray_img, resize)
    equalize_img = cv2.equalizeHist(resize_img)
    
    return equalize_img
