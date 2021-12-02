from flask import Flask, render_template, flash, request, redirect, url_for
import os
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from werkzeug.utils import secure_filename


vgg_model = None
vgg_model_file = "models/vgg_model.sav"
UPLOAD_DIR = os.path.join('static', 'uploads')

app = Flask(__name__)
app.secret_key = "sssp_developers"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
files_to_delete = []
classname = {0:'TC-1',  1:'Classroom Complex',  2:'CS Lab', 3:'Cricket/Football ground', 4:'Front', 5:'Girls hostel', 6:'Guest house', 7:'Gym', 8:'Health center', 9:'Hostel', 10:'Indoor Stadium', 11:'Lab', 12:'Library', 13:'Mess', 14:'OAT steps', 15:'Outdoor courts', 16:'Parking lot', 17:'Roads', 18:'Classroom Complex', 19:'Classroom Complex', 20:'Hostels', 21:'Hostel', 22:'Indoor Stadium', 23:'Lab', 24:'Library', 25:'OAT', 26:'TC-22'}


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def hello():
    if len(files_to_delete) > 5:
        for file in files_to_delete:
            filename = os.path.join("static", "uploads", file)
            if os.path.exists(filename):
                os.remove(filename)
    return render_template("index.html")

@app.route("/", methods = ["POST"])
def submit():
    global vgg_model, vgg_model_file
    if request.method == "POST":
        if 'image' not in request.files:
            flash('No image')
            return redirect(request.url)
        img = request.files['image']
        if img.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if img and allowed_file(img.filename):
            filename = secure_filename(img.filename)
            if not os.path.isdir(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)
            fname = os.path.join(UPLOAD_DIR, filename)
            img.save(fname)
            print("img saved", fname)
            loaded_img = image.load_img(fname, target_size=(224, 224))
            loaded_img = img_to_array(loaded_img)
            img_array = np.array([loaded_img])
            # print("shape", img_array.shape)
            if vgg_model == None:
                vgg_model = pickle.load(open(vgg_model_file, 'rb'))
            result = vgg_model.predict(img_array)
            # print("res", result[0])
            cluster = np.where(result[0] == np.amax(result[0]))[0][0]
            # print("cluster", cluster)
            return render_template("index.html", cluster = classname[cluster], filename= filename)
        else:
            flash('Allowed image types are -> png, jpg, jpeg')
            return redirect(request.url)

@app.route('/display/')
def display_bg():
    return redirect(url_for('static', filename='bg.jpg'), code=301)

@app.route('/display/<filename>')
def display_image(filename):
    files_to_delete.append(filename)
    print("helo")
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


	#print('display_image filename: ' + filename)
    


if __name__ == "__main__":
    app.run(debug=True)