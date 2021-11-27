from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/sub", methods = ["POST"])
def submit():
    imgname = "No Image"
    if request.method == "POST":
        img = request.files['image']
        img.save(os.path.join(img.filename))
        imgname = img.filename
    return render_template("sub.html", arg = imgname)


if __name__ == "__main__":
    app.run(debug=True)