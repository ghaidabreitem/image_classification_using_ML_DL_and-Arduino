import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


with open("svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

with open("svm_pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("X_test_svm.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("y_test_svm.pkl", "rb") as f:
    y_test = pickle.load(f)


X_test_pca = pca.transform(X_test)


y_pred = svm.predict(X_test_pca)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Cat','Dog']))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Cat", "Dog"],
            yticklabels=["Cat", "Dog"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.savefig("svm_confusion_matrix.png")
plt.show()
