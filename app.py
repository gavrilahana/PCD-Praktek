import numpy as np
from PIL import Image
import image_processing
import os
from flask import Flask, render_template, request, make_response, jsonify
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html", file_path="img/image_here.jpg")


@app.route("/about")
@nocache
def about():
    return render_template('about.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_now.jpg")
    copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/normal", methods=["POST"])
@nocache
def normal():
    copyfile("static/img/img_normal.jpg", "static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    image_processing.grayscale()
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/binary", methods=["POST"])
@nocache
def binary():
    img_path = "static/img/img_now.jpg"
    image_processing.binary(img_path)  
    return render_template("uploaded.html", file_path="img/img_now_binary.jpg")


@app.route("/zoomin", methods=["POST"])
@nocache
def zoomin():
    image_processing.zoomin()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/zoomout", methods=["POST"])
@nocache
def zoomout():
    image_processing.zoomout()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_left", methods=["POST"])
@nocache
def move_left():
    image_processing.move_left()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_right", methods=["POST"])
@nocache
def move_right():
    image_processing.move_right()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_up", methods=["POST"])
@nocache
def move_up():
    image_processing.move_up()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_down", methods=["POST"])
@nocache
def move_down():
    image_processing.move_down()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_addition", methods=["POST"])
@nocache
def brightness_addition():
    image_processing.brightness_addition()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_substraction", methods=["POST"])
@nocache
def brightness_substraction():
    image_processing.brightness_substraction()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_multiplication", methods=["POST"])
@nocache
def brightness_multiplication():
    image_processing.brightness_multiplication()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_division", methods=["POST"])
@nocache
def brightness_division():
    image_processing.brightness_division()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/histogram_equalizer", methods=["POST"])
@nocache
def histogram_equalizer():
    image_processing.histogram_equalizer()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/edge_detection", methods=["POST"])
@nocache
def edge_detection():
    image_processing.edge_detection()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/blur", methods=["POST"])
@nocache
def blur():
    image_processing.blur()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/sharpening", methods=["POST"])
@nocache
def sharpening():
    image_processing.sharpening()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/histogram_rgb", methods=["POST"])
@nocache
def histogram_rgb():
    image_processing.histogram_rgb()
    if image_processing.is_grey_scale("static/img/img_now.jpg"):
        return render_template("histogram.html", file_paths=["img/grey_histogram.jpg"])
    else:
        return render_template("histogram.html", file_paths=["img/red_histogram.jpg", "img/green_histogram.jpg", "img/blue_histogram.jpg"])


@app.route("/thresholding", methods=["POST"])
@nocache
def thresholding():
    lower_thres = int(request.form['lower_thres'])
    upper_thres = int(request.form['upper_thres'])
    image_processing.threshold(lower_thres, upper_thres)
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/erosi", methods=["POST"])
@nocache
def erosi():
    image_processing.erosi()
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/dilasi", methods=["POST"])
@nocache
def dilasi():
    image_processing.dilasi()
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/opening", methods=["POST"])
@nocache
def opening():
    image_processing.opening()
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/closing", methods=["POST"])
@nocache
def closing():
    image_processing.closing()
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/count1", methods=["POST"])
@nocache
def count1():
    num_objects=image_processing.count1()
    return render_template("uploaded.html", num_objects=num_objects, file_path="img/img_now.jpg")

@app.route("/count2", methods=["POST"])
@nocache
def count2():
    num_objects=image_processing.count2()
    return render_template("uploaded.html", num_objects=num_objects, file_path="img/img_now.jpg")

@app.route("/count3", methods=["POST"])
@nocache
def count3():
    num_objects=image_processing.count3()
    return render_template("uploaded.html", num_objects=num_objects, file_path="img/img_now.jpg")


@app.route("/create_digit_images", methods=["GET"])
@nocache
def create_digit_images():
    image_processing.create_digit_images()  # Panggil fungsi create_digit_images dari modul image_processing
    return "Digit images created successfully"

@app.route("/extract_contours", methods=["GET"])
@nocache
def extract_contours():
    image_processing.extract_contours()  # Panggil fungsi extract_contours dari modul image_processing
    return "Contours extracted successfully"

@app.route("/freeman_chain_code", methods=["GET"])
@nocache
def freeman_chain_code():
    image_processing.freeman_chain_code()  # Panggil fungsi freeman_chain_code dari modul image_processing
    return "Freeman Chain Code generated successfully"

@app.route("/zhang_suen_thinning", methods=["GET"])
@nocache
def zhang_suen_thinning():
    image_processing.zhang_suen_thinning()  # Panggil fungsi zhang_suen_thinning dari modul image_processing
    return "Thinning completed successfully"

@app.route("/save_to_csv", methods=["GET"])
@nocache
def save_to_csv():
    image_processing.save_to_csv()  # Panggil fungsi save_to_csv dari modul image_processing
    return "Data saved to CSV successfully"

@app.route('/identify_uploaded_image', methods=['POST'])
@nocache
def identify_uploaded_image():
    # Memeriksa apakah ada file gambar yang diunggah
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    # Mengambil file gambar yang diunggah
    uploaded_file = request.files['image']
    
    # Menyimpan file gambar yang diunggah
    uploaded_file.save("static/img/img_now.jpg")
    
    # Mengidentifikasi angka dalam gambar yang diunggah
    identified_digit = image_processing.identify_uploaded_image("static/img/img_now.jpg")
    
    return jsonify({'digit': identified_digit})
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
