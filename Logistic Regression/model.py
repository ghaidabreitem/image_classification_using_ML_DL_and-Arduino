import os
import cv2
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.utils import shuffle


DATA_PATH = r"/\data"
IMG_SIZE = 64


data = []
labels = []

for category in ["cat", "dog"]:
    path = os.path.join(DATA_PATH, category)
    label = 0 if category == "cat" else 1

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # flatten + normalize
            img = img.flatten() / 255.0

            data.append(img)
            labels.append(label)

        except:
            pass

X = np.array(data)
y = np.array(labels)

print("تم تحميل البيانات:", len(X))

X, y = shuffle(X, y, random_state=42)


X = X[:8000]
y = y[:8000]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pca = PCA(n_components=80)
X_train_pca = pca.fit_transform(X_train)


logistic = LogisticRegression(max_iter=1000)
logistic.fit(X_train_pca, y_train)

print("تم تدريب Logistic Regression")


with open("logistic_model.pkl", "wb") as f:
    pickle.dump(logistic, f)

with open("logistic_pca.pkl", "wb") as f:
    pickle.dump(pca, f)


with open("X_test_logistic.pkl", "wb") as f:
    pickle.dump(X_test, f)

with open("y_test_logistic.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("تم حفظ موديل Logistic Regression ✅")
