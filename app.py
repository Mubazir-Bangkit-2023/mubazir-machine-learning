from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.metrics import Metric, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import custom_object_scope
from tensorflow import keras
import tensorflow as tf
from io import BytesIO
import numpy as np
import uvicorn

# Define the custom F1Score class
class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

# Path to the pre-trained model
MODEL_PATH = 'BestModel.h5'

# Load and compile the model with custom_object_scope including the custom F1Score metric
with custom_object_scope({'F1Score': F1Score, 'Precision': Precision, 'Recall': Recall}):
    model = load_model(MODEL_PATH)
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall(), F1Score()])

class_labels = [
    'Fresh Apples', 'Fresh Banana', 'Fresh Cucumber',
    'Fresh Oranges', 'Rotten Apples', 'Rotten Banana',
    'Rotten Cucumber', 'Rotten Oranges',
]

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Load the image from the uploaded file
    contents = await file.read()
    img = load_img(BytesIO(contents), target_size=(299, 299))  # Adjust target size to your model's expected input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Preprocess image as per the model's requirements

    # Predict the class using the loaded model
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Confidence score in percentage

    # Create response data
    kesegaran, jenis_buah = class_labels[predicted_class_index].split(' ')

    response_data = {
        "Kesegaran": kesegaran,
        "Jenis_Buah": jenis_buah,
        "Confidence": f"{confidence:.2f}%" if confidence else None
    }

    
    return JSONResponse(content=response_data)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

