import os
import uvicorn
import numpy as np
import traceback
from io import BytesIO

from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from utils import load_image_into_numpy_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = FastAPI()

@app.post("/predict_image")
async def predict_image(img: UploadFile, response: Response):
    try:
        # Checking if it's an image
        if img.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"

        # Read file content
        file_content = await img.read()

        # Convert bytes to a file-like object
        file_like_object = BytesIO(file_content)

        # Load the image from the file-like object
        img = load_img(file_like_object, target_size=(150, 150))

        # Load the model
        model = load_model('combined_model.h5')

        # Prepare the image for prediction
        image = img_to_array(img)
        image = np.expand_dims(image, axis=0)

        # Predict the class of the image
        prediction = model.predict([img_array, np.zeros((1, 150, 150, 3))])[0]
        confidence = np.max(prediction)
        print(f"Predictions: {prediction}, Confidence: {confidence}")  # Debugging output
        predicted_class_index = np.argmax(prediction)
        if predicted_class_index >= len(class_labels) or confidence < confidence_threshold:
            return "Cannot be predicted", confidence
        predicted_label = class_labels[predicted_class_index]
        return predicted_label, confidence

# Menentukan label

        return {"result":predicted_label}
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return str(e)
