import serial
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model


arduino = serial.Serial('COM6', 9600)
time.sleep(2)



model = load_model(r"/\code\CNN\cnn_cats_dogs_cleaned.h5")
IMG_SIZE = 128

predictions = []
camera_open = True
cap = cv2.VideoCapture(0)

cv2.namedWindow("Camera")
print("Camera started automatically.")
print("Press 'q' to show current Final Prediction, 's' to resume camera, 'e' to exit program.")

while True:
    if camera_open:
        ret, frame = cap.read()
        if not ret:
            continue


        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img_input = np.expand_dims(img, axis=0)


        pred = model.predict(img_input, verbose=0)
        class_idx = np.argmax(pred)
        label = "Cat" if class_idx == 0 else "Dog"
        predictions.append(label)


        if class_idx == 0:
            arduino.write(b'C')
        else:
            arduino.write(b'D')


        cv2.putText(frame, label, (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)
        cv2.imshow("Camera", frame)
        time.sleep(0.3)


    key = cv2.waitKey(50) & 0xFF

    if key == ord('q'):

        camera_open = False
        if predictions:
            final_prediction = max(set(predictions), key=predictions.count)
            print("Current Final Prediction:", final_prediction)

            pause_frame = np.zeros((200,400,3), dtype=np.uint8)
            cv2.putText(pause_frame, f"Final Prediction: {final_prediction}",
                        (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Camera", pause_frame)

    elif key == ord('s') and not camera_open:

        cap = cv2.VideoCapture(0)
        camera_open = True
        print("Camera resumed...")
    elif key == ord('e'):
        break


if camera_open:
    cap.release()
cv2.destroyAllWindows()


if predictions:
    final_prediction = max(set(predictions), key=predictions.count)
    print("Final Prediction:", final_prediction)
    if final_prediction == "Cat":
        arduino.write(b'C')
    else:
        arduino.write(b'D')
    time.sleep(1)
    arduino.write(b'0')

arduino.close()
