#!/usr/bin/env python3
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from bottle import route, post, static_file, run, request
from tes import *


@post("/upload")
def submit_image():
    print("HEy")
    # do some cleanup
    if os.path.isfile("upload.png"):
        os.remove("upload.png")

    # get the file
    upload = request.files.get('fileToUpload')
    name, extension = os.path.splitext(upload.filename)
    
    # save the file
    upload_filename = "upload" + extension
    upload.save(upload_filename)

    # lets do detection
    do_processing(upload_filename)

    return static_file("output.jpg", root="./")


@route("/")
def home_page():
    return static_file("home.html", root="")


# run the server
run(host="0.0.0.0", port=9000, debug=True)
