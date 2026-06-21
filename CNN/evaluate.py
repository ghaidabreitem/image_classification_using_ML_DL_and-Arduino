import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = 128
BATCH_SIZE = 32
DATA_PATH = r"/\data"


datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


model = tf.keras.models.load_model("cnn_cats_dogs_cleaned.h5")


y_pred_prob = model.predict(test_generator)
y_pred = y_pred_prob.argmax(axis=1)
y_true = test_generator.classes


accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(test_generator.class_indices.keys()),
            yticklabels=list(test_generator.class_indices.keys()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN Confusion Matrix")
plt.savefig("cnn_confusion_matrix.png")
plt.show()
