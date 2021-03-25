from flask import Flask, make_response, jsonify
import json
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image

app = Flask(__name__)

@app.route('/', methods = ["GET"])
def predict():
    # load model
    model = load_model('./models/epochs20.h5')
    
    img_path = "./images/no.jpg"
    img = image.load_img(img_path, target_size=(224, 224))

    img_array = image.img_to_array(img)
    new_array = img_array.reshape(1,224,224,3)
    prediction = model.predict(new_array)

    #####################################################
    ####         status = 1 means accident           ####
    ####         status = 0 means noAccident         ####
    #####################################################

    data = {"status" : 0 }
    if(prediction[0] < 0.5) :
        print("Accident")
        data["status"] = 1
    else:
        print("No Accident Yet")

    response = make_response(
                jsonify(
                    data
                ),
                200,
            )
    response.headers["Content-Type"] = "application/json"

    return response

if __name__ == '__main__':
    app.run()