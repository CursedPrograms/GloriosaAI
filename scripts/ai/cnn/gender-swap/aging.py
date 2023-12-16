import cv2
import numpy as np

def age_face(image_path):
    image = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    age_model = lambda x: int(1.1 * x + 10)
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        age = age_model(w)
        face_roi = cv2.resize(face_roi, (w, h))
        face_roi = cv2.GaussianBlur(face_roi, (101, 101), 30)
        image[y:y+h, x:x+w] = cv2.addWeighted(face_roi, 0.5, image[y:y+h, x:x+w], 0.5, 0)

    return image
input_image_path = 'path/to/your/input/image.jpg'
output_image = age_face(input_image_path)

cv2.imshow('Aged Face Simulation', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
