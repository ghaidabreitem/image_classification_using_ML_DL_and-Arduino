import serial
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model


arduino = serial.Serial('COM6', 9600)
time.sleep(2)


model = load_model(r"/\code\CNN\cnn_cats_dogs_cleaned.h5")
IMG_SIZE = 128


cap = cv2.VideoCapture(0)
final_prediction = None  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img_input = np.expand_dims(img, axis=0)

    pred = model.predict(img_input, verbose=0)
    class_idx = np.argmax(pred)

    if class_idx == 0:
        label = "Cat"
        arduino.write(b'C')
    else:
        label = "Dog"
        arduino.write(b'D')

    final_prediction = label


    cv2.putText(frame, label, (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\nFinal Prediction:", final_prediction)


time.sleep(1)


cap.release()
cv2.destroyAllWindows()
arduino.close()
