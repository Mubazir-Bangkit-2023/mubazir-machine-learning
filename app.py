from fastapi import FastAPI, UploadFile, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from io import BytesIO
import json

app = FastAPI()
model = load_model('combined_model_new.h5')
class_labels = ['Apel', 'Pisang', 'Paprika', 'Jeruk', 'Wortel', 'Timun']

@app.post("/predict_image")
async def predict_image(img: UploadFile, response: Response):
    try:
        if img.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="File is not an image")

        # Read file content and prepare image
        file_content = await img.read()
        file_like_object = BytesIO(file_content)
        img = load_img(file_like_object, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        prediction = model.predict([img_array, np.zeros((1, 150, 150, 3))])[0]
        confidence = np.max(prediction)
        predicted_class_index = np.argmax(prediction)

        if predicted_class_index >= len(class_labels) or confidence < 0.6:
            result = {"result": "Cannot be predicted", "confidence": int(confidence*100), }
        else:
            predicted_label = class_labels[predicted_class_index]
            result = {"result": predicted_label, "confidence": int(confidence*100),}

        # Convert result using jsonable_encoder to ensure JSON compatibility
        return jsonable_encoder(result)

    except Exception as e:
        traceback.print_exc()
        return Response(content=str(e), status_code=500)


