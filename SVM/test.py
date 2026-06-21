import serial
import time
import cv2
import numpy as np
import pickle


arduino = serial.Serial('COM6', 9600)
time.sleep(2)


with open("svm_model.pkl", "rb") as f:
    svm = pickle.load(f)
with open("svm_pca.pkl", "rb") as f:
    pca = pickle.load(f)

IMG_SIZE = 64
predictions = []
camera_open = True


cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera")
print("Camera started automatically.")
print("Press 'q' to pause, 's' to resume, 'e' to exit.")

while True:
    if camera_open:
        ret, frame = cap.read()
        if not ret:
            continue


        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.flatten() / 255.0
        img_input = pca.transform([img])  # تحويل PCA


        class_idx = svm.predict(img_input)[0]
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
        # Pause camera
        camera_open = False
        if predictions:
            final_prediction = max(set(predictions), key=predictions.count)
            print("Current Final Prediction:", final_prediction)
            pause_frame = np.zeros((200,400,3), dtype=np.uint8)
            cv2.putText(pause_frame, f"Final Prediction: {final_prediction}",
                        (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Camera", pause_frame)

    elif key == ord('s') and not camera_open:
        # Resume camera
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
