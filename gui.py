import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.preprocessing import image

# Create a dictionary of models and their names
models = {
    "ResNet50": ResNet50(weights="imagenet"),
    "VGG16": VGG16(weights="imagenet"),
    "MobileNet": MobileNet(weights="imagenet"),
    "InceptionV3": InceptionV3(weights="imagenet"),
    "Xception": Xception(weights="imagenet")
}

# Define a function to preprocess the image
def preprocess_image(image, model_name):
    if model_name == "ResNet50":
        # Resize the image to the required input size of the ResNet50 model
        image = image.resize((224, 224))
        # Preprocess the image using the ResNet50 model's preprocessing function
        image_array = np.array(image)
        image_array = preprocess_input(image_array)
    elif model_name == "InceptionV3":
        # Resize the image to the required input size of the InceptionV3 model
        image = image.resize((299, 299))
        # Preprocess the image using the InceptionV3 model's preprocessing function
        image_array = np.array(image)
        image_array = tf.keras.applications.inception_v3.preprocess_input(image_array)
    else:
        raise ValueError("Invalid model name: {}".format(model_name))
    # Add an extra dimension to represent the batch size of 1
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define a function to predict the label of the image
def predict_label(image_path, model_name):
    # Load the model
    model = models[model_name]
    # Load the image
    image = Image.open(image_path)
    # Preprocess the image
    image_array = preprocess_image(image, model_name)
    # Predict the label
    predictions = model.predict(image_array)
    # Decode the predictions using the model's decoding function
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    # Get the label name of the highest predicted class
    label_name = decoded_predictions[0][1]
    return label_name

# Define a function to open the file dialog and select an image
def select_image():
    # Open the file dialog
    file_path = filedialog.askopenfilename()
    # Display the selected image in the GUI
    image = Image.open(file_path)
    photo = ImageTk.PhotoImage(image)
    canvas.itemconfigure(image_item, image=photo)
    canvas.image = photo
    # Store the file path for later prediction
    global selected_image_path
    selected_image_path = file_path

# Define a function to predict the label of the selected image
def predict_selected_image():
    # Get the selected model name from the combo box
    model_name = model_var.get()
    # Predict the label of the selected image using the selected model and display it in the GUI
    label = predict_label(selected_image_path, model_name)
    label_var.set(label)

# Create the GUI window
root = tk.Tk()
root.title("Image Name Tagging Predictor")

# Create a combo box to select the model
model_var = tk.StringVar(value="ResNet50")
model_label = tk.Label(root, text="Select Model:")
model_label.pack(side=tk.TOP, padx=10, pady=10)
model_box = tk.OptionMenu(root, model_var, *models.keys())
model_box.pack()

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack(side=tk.TOP, padx=10, pady=10)

# Create a button to select the image
image_path = tk.StringVar()
def select_image():
    path = filedialog.askopenfilename()
    image_path.set(path)
    # Update the image label with the selected image
    img = Image.open(image_path.get())
    img = img.resize((400, 400))
    img = ImageTk.PhotoImage(img)
    image_label.configure(image=img)
    image_label.image = img # Keep a reference to avoid garbage collection

select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(side=tk.BOTTOM, pady=10)

# Create a button to predict the label
def predict_label():
    # Load the selected model
    selected_model = models[model_var.get()]

    # Preprocess the image
    img = image.load_img(image_path.get(), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make the prediction
    preds = selected_model.predict(x)
    label = decode_predictions(preds, top=1)[0][0][1]

    # Display the prediction
    prediction_label.config(text=f"Predicted label: {label}")

predict_button = tk.Button(root, text="Predict Label", command=predict_label)
predict_button.pack(side=tk.BOTTOM, pady=10)

# Create a label to display the prediction
prediction_label = tk.Label(root, text="Predicted label: ")
prediction_label.pack(side=tk.BOTTOM, padx=10, pady=10)
root.mainloop()




