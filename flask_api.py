
import flask
from flask import request, jsonify
import pickle
import json
import tensorflow
#import keras_segmentation
#from keras_segmentation.models.unet import vgg_unet
import cv2
import numpy as np
import PIL
from PIL import Image, ImageOps, ImageFilter


# API setup
app_flask = flask.Flask(__name__)
app_flask.config["DEBUG"] = True

# Create URL of Main Page:  http://127.0.0.1:5000 IF locally
@app_flask.route("/", methods=["GET"])
def home():
    return "<h1>FLASK PREDICTION API</h1><p>ENDPOINT is under the /predict route.</p>"


# Create URL of Predict Endpoint:http://127.0.0.1:5000/predict IF locally
@app_flask.route("/predict", methods=["POST"])
def predict_image():

    # Get the JSON data from the request
    data_as_json = request.get_json()

    # De-Serialize INTO A NUMPY ARRAY the JSON received !!!!
    image_as_array = pickle.loads(json.loads(data_as_json).encode('latin-1'))

    # Load the existing Keras SAVED model in order to GET ITS "INPUT SHAPE" REQUIREMENTS:
    new_model_keras_v2 = tensorflow.keras.models.load_model("model_keras_with_generator")
    input_height = new_model_keras_v2.input_shape[1]
    input_width = new_model_keras_v2.input_shape[2]

    # Process the image to be Segmented (now it is in an ARRAY format!!!!)
    # RESIZE the Image (in array format) into "THIS KERAS MODEL" REQUIRED input FORMAT
    input_data = cv2.resize(image_as_array, (input_width, input_height))
    # CONVERT Image to array
    #input_data = np.array(image)

    # Actual Prediction stage: Probabilites MATRIX that EACH pixel belongs to each CATEGORY
    predicted_segmentation_probas = new_model_keras_v2.predict(np.expand_dims(input_data, axis=0))
    # CONVERT PREDICTED PROBA TO INDEX INTERGERS
    predicted_segmentation_probas = np.squeeze(predicted_segmentation_probas, axis=0)
    predicted_segmentation_pixels_2 = np.argmax(predicted_segmentation_probas, axis=1)

    # Reshape the array containng the PREDICTED MASK PIXELS
    predicted_segmentation_pixels_2 = predicted_segmentation_pixels_2.reshape(int(input_height / 2), int(input_width / 2))
    # Convert pixels ARRAY to "uint8" format
    predicted_segmentation_pixels_2 = predicted_segmentation_pixels_2.astype(np.uint8)

    # Create an Image from the pixel array
    predicted_segmentation_image_2 = Image.fromarray(predicted_segmentation_pixels_2)
    shape_of_predicted = predicted_segmentation_image_2.size

    # IMAGE: Serialize as JSON the predicted mask
    image_as_array = np.array(predicted_segmentation_image_2)
    image_as_json = json.dumps(pickle.dumps(image_as_array).decode("latin-1"))

    # ARRAY: Serialize as JSON the predicted mask
    #predicted_mask_as_json = json.dumps(pickle.dumps(predicted_segmentation_pixels_2).decode("latin-1"))

    #response = {"Predicted Mask": predicted_segmentation_pixels_2}

    #return jsonify(response=predicted_mask_as_json, message = shape_of_predicted)
    return jsonify(response=image_as_json, message=shape_of_predicted)
    #return {"response":image_as_json, "message":shape_of_predicted}


#************Launch API LOCALLY*************
app_flask.run()
#app_flask.run(port=5000)

