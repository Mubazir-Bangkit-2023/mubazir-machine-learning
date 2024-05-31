from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

app = FastAPI()

# Load the model

combined_model = load_model('combined_model_new.h5')
class_labels = ['Apel', 'Pisang', 'Paprika', 'Jeruk', 'Wortel', 'Timun']

def predict_jenis_buah(model, img_array, class_labels, confidence_threshold=0.6):
    prediction = model.predict([img_array, np.zeros((1, 150, 150, 3))])[0]
    confidence = np.max(prediction)
    predicted_class_index = np.argmax(prediction)
    if predicted_class_index >= len(class_labels) or confidence < confidence_threshold:
        return "0", confidence
    predicted_label = class_labels[predicted_class_index]
    return predicted_label, confidence

def predict_kesegaran(model, img_array):
    prediction = model.predict([np.zeros((1, 224, 224, 3)), img_array])[1]
    return 'Segar' if prediction[0] > 0.01 else 'Tidak Segar'

@app.post("/predict_image")
async def predict_image(file: UploadFile):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Read image file and prepare it
    contents = await file.read()
    img = load_img(BytesIO(contents), target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict jenis buah
    jenis_buah, confidence = predict_jenis_buah(combined_model, img_array, class_labels)
    if jenis_buah == "0":
        kesegaran_buah = "0"
    else:
        img = load_img(BytesIO(contents), target_size=(150, 150))
        img_array_freshness = img_to_array(img)
        img_array_freshness = np.expand_dims(img_array_freshness, axis=0) / 255.0
        kesegaran_buah = predict_kesegaran(combined_model, img_array_freshness)

    # Construct result
    result = {
        'Jenis Buah': jenis_buah,
        'Confidence': int(confidence*100),
        'Kesegaran Buah': kesegaran_buah
    }

    return JSONResponse(content=result)


