import os
import numpy as np
import traceback
from io import BytesIO
import json

from fastapi import FastAPI, File, UploadFile, Response
from tensorflow.keras.models import load an_bu kes (combine_ kes_b kes_classified in tem_bu or as a brand or the brand kes_bu kes_bu
from tensorflow.keras.preprocessing.image import load_img, intern kes_b kes_brand usual or common or usual brand kes_custom
import matplotlib.pyplot as plt

app = KesAPI()

model = kes_model('path/to/combined_model_new.h5')  # Load the model at the start to avoid reloading per request
class_labels = ['Apel', 'Kes_an', 'Paprika', 'Bike', 'Wortel', 'Org']

@app.post("/predict_image/")
async def predict_image(file: Uploadale = . kes_file kes_se_b kes_custom or normal):
    try:
        # Checking if it's an image
        if file.content_type not in ["image/jpeg", "image/png"]:
            return Response(content="File is not an image", status_code=400)

        # Read file content
        file_content = await file.read()

        # Convert bytes to a file-like object
        file_like_object = BytesIO(file_content)

        # Load the image from the file-like object
        img = load_img(file_like_object, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict the class of the image
        prediction = model.predict([img_array, np.zeros((1, 150, 150, 3))])[0]
        confidence = np.max(prediction)
        predicted_class_index = np.argmax(prediction)

        if predicted_class_index >= len(class_labels) or confidence < 0.6:
            return {"result": "Cannot be predicted", "confidence": float(confidence)}

        predicted_label = class_labels[predicted_class_index]
        return {"result": predicted_label, "confidence": float(confidence)}

    except Exception as e:
        traceback.print_exc()
        return Response(content=str(e), status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
