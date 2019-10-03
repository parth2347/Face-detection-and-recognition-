# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 23:17:07 2019

@author: HP
"""
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os
import numpy as np
import cv2

classes = ["",'Parth','Nevil']

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('D:\python\haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder):
    dirs = os.listdir(data_folder)
    
    faces = []
    labels = []
    
    for dir_name in dirs:
        label = int(dir_name.replace("s",""))
        sub_dir_path = data_folder + "/"+dir_name
        
        sub_images = os.listdir(sub_dir_path)
        
        for image_name in sub_images:
            
            image_path = sub_dir_path + "/"+ image_name
            
            image = cv2.imread(image_path)
            
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            
            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels
    
print("Preparing data...")
faces, labels = prepare_training_data("training")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
        
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))

def predict(test_img):
    img = test_img.copy()
    
    face, rect = detect_face(img)

    label= face_recognizer.predict(face)
    print('label_1',label)
    label_text = classes[int(label[0])]
    #label_text = classes[label]
    
    print(label_text)
    draw_rectangle(img, rect)
    
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img

print("Predicting images...!!!!!!!!!!!!!")


test_img1 = cv2.imread(r'D:\python\p2.jpg')
test_img2 = cv2.imread(r'D:\python\n2.jpg')

 
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)

print("Prediction complete")

WIDTH = 1000
HEIGHT = 1000

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow(classes[1], predicted_img1)
cv2.resizeWindow(classes[1], WIDTH, HEIGHT)

#cv2.imshow(classes[1], predicted_img1)
cv2.imshow(classes[2], predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()



        
    
