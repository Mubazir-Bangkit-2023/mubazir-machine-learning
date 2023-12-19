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
        model = load_model('model.h5')

        # Prepare the image for prediction
        image = img_to_array(img)
        image = np.expand_dims(image, axis=0)

        # Predict the class of the image
         arr = model.predict(image)
          if arr[0][0]==1:
            labels='Fresh Apples'
          elif arr[0][1]==1:
            labels='Fresh Banana'
          elif arr[0][2]==1:
            labels='Fresh Oranges'
          elif arr[0][3]==1:
            labels='Rotten Apples'
          elif arr[0][4]==1:
            labels='Rotten Banana'
          elif arr[0][5]==1:
            labels='Rotten Oranges'
              
        return {"result":labels}
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return str(e)
