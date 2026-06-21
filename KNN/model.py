import os
import cv2
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


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

            img = img.flatten() / 255.0

            data.append(img)
            labels.append(label)

        except:
            pass

X = np.array(data)
y = np.array(labels)

print("تم تحميل البيانات:", len(X))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pca = PCA(n_components=100)
X_train = pca.fit_transform(X_train)


params = {'n_neighbors': [3,5,7]}

grid = GridSearchCV(KNeighborsClassifier(), params, cv=3)
grid.fit(X_train, y_train)

knn = grid.best_estimator_

print("أفضل K:", grid.best_params_)


with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)

with open("X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("تم حفظ الموديل والبيانات")
