import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)


X_test = pca.transform(X_test)


y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Cat", "Dog"],
            yticklabels=["Cat", "Dog"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")

plt.savefig("knn_confusion_matrix.png")
plt.show()
