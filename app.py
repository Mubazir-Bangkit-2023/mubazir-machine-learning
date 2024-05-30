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
        # img = load_img(file_like_object, target_size=(150, 150))

        # Load the model
        model = load_model('combined_model_new.h5')

        # Prepare the image for prediction
        # image = img_to_array(img)
        # image = np.expand_dims(image, axis=0)

        # Predict the class of the image
class_labels = ['Apel', 'Pisang', 'Paprika', 'Jeruk', 'Wortel', 'Timun']

def predict_jenis_buah(model, class_labels, file_name, target_size=(224, 224), confidence_threshold=0.6):
    img = load_img(file_name, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict([img_array, np.zeros((1, 150, 150, 3))])[0]
    confidence = np.max(prediction)
    predicted_class_index = np.argmax(prediction)
    if predicted_class_index >= len(class_labels) or confidence < confidence_threshold:
        return "0", confidence
    predicted_label = class_labels[predicted_class_index]
    return predicted_label, confidence

def predict_kesegaran(model, file_name, target_size=(150, 150)):
    img = load_img(file_name, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict([np.zeros((1, 224, 224, 3)), img_array])[1]
    predicted_label = 'Segar' if prediction[0] > 0.01 else 'Tidak Segar'
    return predicted_label

# Upload and predict new images
uploaded = files.upload()
results = {}
for fn in uploaded.keys():
    file_path = fn
    jenis_buah, confidence = predict_jenis_buah(combined_model, class_labels, file_path)
    if jenis_buah == "0":
        kesegaran_buah = "0"
    else:
        kesegaran_buah = predict_kesegaran(combined_model, file_path)

    # Store the results in a dictionary
    results[fn] = {
        'Jenis Buah': jenis_buah,
        'Confidence': float(confidence),
        'Kesegaran Buah': kesegaran_buah
    }

    # Optionally, display the image and the predictions
    img = load_img(file_path)
    plt.imshow(img)
    plt.title(f'Predicted: {jenis_buah} (Confidence: {confidence:.2f}), Kesegaran: {kesegaran_buah}')
    plt.axis('off')
    plt.show()

# Convert the results to JSON and print/display
json_output = json.dumps(results, indent=4)

# Menentukan label

        return {"result":json_output}
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return str(e)
