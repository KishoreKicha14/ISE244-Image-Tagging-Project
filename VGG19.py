
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
#import seaborn as sns
import matplotlib.pyplot as plt

def load_images(folder_path, label, batch_size, IMG_SIZE):
    images = []
    labels = []
    for i, filename in enumerate(os.listdir(folder_path)):
        img = cv2.imread(os.path.join(folder_path, filename))
        print(img.shape)
        height, width, _ = img.shape
        aspect_ratio = width / height
        new_width = int(aspect_ratio * IMG_SIZE)
        img = cv2.resize(img, (new_width, IMG_SIZE))
        if new_width > IMG_SIZE:
            start = (new_width - IMG_SIZE) // 2
            img = img[:, start:start+IMG_SIZE, :]
        else:
            padding = ((0, 0), ((IMG_SIZE - new_width) // 2, (IMG_SIZE - new_width) // 2), (0, 0))
            img = np.pad(img, padding, mode="constant")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape != (IMG_SIZE, IMG_SIZE, 3):
            print(f"Image {filename} has unexpected shape {img.shape}")
            continue
        images.append(img)
        labels.append(label)
        if (i + 1) % batch_size == 0:
            print(len(np.array(images)))
            yield np.array(images), np.array(labels)
            images = []
            labels = []
    if images and labels:
        yield np.array(images), np.array(labels)




# Define the parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
folder_paths = ["/Users/kishorekumaar/Documents/SJSU/ISE244/ISE/ISE PROJECT/beach", "/Users/kishorekumaar/Documents/SJSU/ISE244/ISE/ISE PROJECT/palace", "/Users/kishorekumaar/Documents/SJSU/ISE244/ISE/ISE PROJECT/barn"]
num_classes = len(folder_paths)

# Define the data generator with augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

# Define the model architecture
model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Flatten(),
    layers.Dense(4096, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(4096, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])


# Load the images and their corresponding labels
images = []
labels = []
for i, folder_path in enumerate(folder_paths):
    for batch_images, batch_labels in load_images(folder_path, i, BATCH_SIZE, IMG_SIZE):
        images.append(batch_images)
        labels.append(batch_labels)
x = np.concatenate(images)
y = np.concatenate(labels)

# Convert `y_train` to an array
y = np.array(y)

# Convert `y_train` to one-hot encoded vectors
y = to_categorical(y)

# Verify the shape of `y_train`
print(y.shape)
print(y[0])
# Split the dataset into training, validation, and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)




# Use the image data generator to augment the training set
train_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(train_generator, epochs=EPOCHS, validation_data=(x_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
model.save("VGG16.h5")

# Generate predictions for the test set
y_pred = model.predict(x_test)

# Convert the predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate precision, recall, and F1 score for each class
from sklearn.metrics import classification_report
print(classification_report(y_test_labels, y_pred_labels))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Predict on the test set
y_pred = model.predict(x_test)

# Get the predicted labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Get the true labels
y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred_labels)

# Compute the ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred_labels)
roc_auc = auc(fpr, tpr)

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap=plt.cm.Blues)
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(folder_paths)
ax.set_yticklabels(folder_paths)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(num_classes):
    for j in range(num_classes):
        text = ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_title("Confusion matrix")
fig.tight_layout()
plt.savefig("confusion_matrix.png")

# Plot the ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic")
ax.legend(loc="lower right")
fig.tight_layout()
plt.savefig("roc_curve.png")
