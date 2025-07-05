# AI-Image-Classification-Python

# 🧠 Image Classification using Teachable Machine and Python

## 1. Project Title
**Image Recognition with Teachable Machine and Keras**

This project demonstrates how to train an image classification model using Google’s [Teachable Machine](https://teachablemachine.withgoogle.com/), then export and use it with Python and Keras (TensorFlow backend) to classify new input images.

## 2. Description
This beginner-level machine learning project shows how to:
- Collect and train images in Teachable Machine.
- Export the trained model in TensorFlow → Keras format.
- Use a Python script to load the model and classify a new image.
It is designed to introduce students and hobbyists to the workflow of ML inference without deep coding.

## 3. Purpose
- Practice using pre-trained image classification models.
- Understand how to use Keras and Pillow in Python.
- Bridge web-based training tools (Teachable Machine) with local Python scripts.
- Run simple AI inference tasks on custom images.

## 4. Requirements
To run the Python prediction script, you need:
- Python 3.x
- TensorFlow (`pip install tensorflow`)
- Pillow (`pip install pillow`)
- A trained model from Teachable Machine (`keras_model.h5`, `labels.txt`)
- Input image (e.g., `test.jpg`)

## 5. Components
| Item                          | Description                                  |
|-------------------------------|----------------------------------------------|
| Teachable Machine             | Used to create and export the ML model       |
| keras_model.h5                | Trained model file                           |
| labels.txt                    | List of class labels                         |
| Python + Keras + Pillow       | Required libraries for running inference     |
| test image                    | Image to classify                            |

## 6. How the Model Was Trained
1. Visited [Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Selected **Image Project > Standard Image Model**.
3. Created two or more classes (e.g., Apple, Banana).
4. Collected or uploaded training images.
5. Clicked **Train Model**.
6. Exported the model: `Export Model → TensorFlow → Keras`.

## 7. How It Works
- The script loads the model and labels.
- An image is resized to 224x224 and normalized as per model requirements.
- The model predicts the class of the image.
- The output shows the predicted class and the confidence score.

## 8. How to Run
1. Place `keras_model.h5`, `labels.txt`, and the image (e.g., `test.jpg`) in the same folder as the script.
2. Install the required Python libraries if you haven’t already:
pip install tensorflow pillow
3. Open a terminal or command prompt, navigate to the project folder, and run the script:
 python predict.py
4. The script will load the image, process it, and output the predicted class and confidence score, for example:
  Class: Mango  
Confidence Score: 0.9992
![Class Mango](https://github.com/user-attachments/assets/0f956a34-aeec-4e4b-884b-c0dc09549123)




## 9.Python Prediction Script
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps      # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("<IMAGE_PATH>").convert("RGB")

# Resize the image to 224x224 and center crop
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predict the image class
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Output the result
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
