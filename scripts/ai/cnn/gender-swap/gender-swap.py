import cv2
import dlib
import numpy as np

def gender_swap(image_path):
    image = cv2.imread(image_path)
    predictor_path = "shape_predictor_68_face_landmarks.dat" 
    predictor = dlib.shape_predictor(predictor_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    def gender_model(face_landmarks):
        return 'female' if face_landmarks[29][1] - face_landmarks[33][1] < 0 else 'male'
    for (x, y, w, h) in faces:
        rect = dlib.rectangle(x, y, x+w, y+h)
        landmarks = predictor(image, rect)
        landmarks = [(p.x, p.y) for p in landmarks.parts()]
        gender = gender_model(landmarks)
        overlay_path = f"overlay_{gender}.png" 
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        overlay = cv2.resize(overlay, (w, h))
        image[y:y+h, x:x+w] = overlay[:, :, :3] * (overlay[:, :, 3:] / 255.0) + image[y:y+h, x:x+w] * (1.0 - overlay[:, :, 3:] / 255.0)

    return image
input_image_path = 'path/to/your/input/image.jpg'
output_image = gender_swap(input_image_path)

cv2.imshow('Gender Swapped Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
