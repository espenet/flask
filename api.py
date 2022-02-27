# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 23:09:12 2022

@author: Khalil
"""
from keras.models import load_model, model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 

from flask import *  
app = Flask(__name__)  
 
@app.route('/')  
def upload():  
    return render_template("upload.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        print(f.filename)
        json_file = open('architecture_olives.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("olivemodel.h5")
        print("Loaded model from disk")
        class_labels = ["normal","rotten"]
        single_image_data = f.filename
        image = load_img(single_image_data, target_size=(224, 224)) 
        from keras.preprocessing.image import img_to_array
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        yhat = loaded_model.predict(image)    
        pred = np.argmax(yhat, axis=-1)
        print(class_labels[pred[0]])

        return render_template("success.html", name = f.filename)
        
  
if __name__ == '__main__': 



    app.run(debug = False)  
