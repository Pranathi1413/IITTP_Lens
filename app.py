from flask import Flask, render_template, request
import os
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

vgg_model = None
vgg_model_file = "models/vgg_model.sav"

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/sub", methods = ["POST"])
def submit():
    global vgg_model, vgg_model_file
    imgname = "No Image"
    if request.method == "POST":
        img = request.files['image']
        if not os.path.isdir("client_images"):
            os.mkdir("client_images")
        fname = os.path.join("client_images", img.filename)
        img.save(fname)
        loaded_img = image.load_img(fname, target_size=(224, 224))
        loaded_img = img_to_array(loaded_img)
        img_array = np.array([loaded_img])
        # print("shape", img_array.shape)
        imgname = img.filename
        if vgg_model == None:
            vgg_model = pickle.load(open(vgg_model_file, 'rb'))
        result = vgg_model.predict(img_array)
        # print("res", result[0])
        cluster = np.where(result[0] == np.amax(result[0]))[0][0]
        # print("cluster", cluster)
    return render_template("sub.html", arg = cluster)


if __name__ == "__main__":
    app.run(debug=True)