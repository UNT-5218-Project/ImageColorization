import os
from flask import Flask, flash, request, redirect, render_template, url_for
import keras
import tensorflow as tf
import numpy as np
from PIL import Image as PImage

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['IMAGES_DIRECTORY'] = './images_directory/'
app.config['STATIC_DIRECTORY'] = './static/'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

original_width = 0
original_height = 0

uploaded_image = os.path.join(app.config['IMAGES_DIRECTORY'], 'uploadedImage.png')

# for model
resized_image = os.path.join(app.config['IMAGES_DIRECTORY'], 'resizedImage.png')
gray_scale_image = os.path.join(app.config['IMAGES_DIRECTORY'], 'grayScaleImage.png')

# for display on website
uploaded_image_for_display = os.path.join(app.config['STATIC_DIRECTORY'], 'uploadedImageForDisplay.png')
gray_scale_image_for_display = os.path.join(app.config['STATIC_DIRECTORY'], 'grayScaleImageForDisplay.png')


# trained_model_path = './trained_model'
# model = keras.models.load_model(trained_model_path)

def allowed_file_formats(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def perform_prediction(uploaded_file):
#     upoaded_image = PImage.open(uploaded_file)
#     new_width = 75
#     new_height = 75
#     resized_image = upoaded_image.resize((new_width, new_height), PImage.ANTIALIAS)
#     img.save('')
#
#     img_array = keras.preprocessing.image.img_to_array(upoaded_image)
#     img_array = tf.expand_dims(img_array, 0) # Create a batch
#     # predicted_output = model.predict(img_array)
#     # classification_id = np.argmax(predicted_output, axis=1)[0]
#     # classification_type = classificationId_to_classificationName(classification_id)
#     return "test"

def perform_resize(uploaded_file):
    global original_width
    global original_height
    uploaded_file = PImage.open(uploaded_file)
    uploaded_file.save(uploaded_image_for_display)
    original_width, original_height = uploaded_file.size
    new_width = 75
    new_height = 75
    resized_file = uploaded_file.resize((new_width, new_height), PImage.ANTIALIAS)
    resized_file.save(resized_image)
    return


def convert_to_grayscale(input_file):
    PImage.open(input_file).convert('L').save(gray_scale_image)
    return


def convert_to_grayscale_for_display(input_file):
    PImage.open(input_file).convert('L').save(gray_scale_image_for_display)
    return


@app.route("/")
def fileFrontPage():
    return render_template('fileform.html')


@app.route('/handleUpload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file_formats(file.filename):
            file.save(uploaded_image)
            perform_resize(uploaded_image)
            convert_to_grayscale(resized_image)

            convert_to_grayscale_for_display(uploaded_image_for_display)

            return render_template('fileform.html', uploaded_image_for_display='uploadedImageForDisplay.png',
                                   grayscaled_image_for_display='grayScaleImageForDisplay.png')
        else:
            return 'Invalid file format'
    return


if __name__ == '__main__':
    app.run()
