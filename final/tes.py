
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from termcolor import colored
import operator



def do_processing(path_to_file):
    args = {"model": "DevaModel.h5","image" : path_to_file}
    labels = ["yna","taa","thaa","daa","dhaa","dna","ta","tha","da","dha","ka","na","pa","pha","ba","bha","ma","yaw","ra","la","waw","kha","saw","sha","sa","ha","ksha","tra","gya","ga","gha","kna","cha","chha","ja","jha","0","1","2","3","4","5","6","7","8","9"]
    lab=[u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',
u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','ksha','tra','gya',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']
  
    image = cv2.imread(args["image"])
    orig = image.copy()
	 
	# pre-process the image for classification
    
    image = cv2.resize(image,(32,32))
   
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
	#print image.shape
    image = np.expand_dims(image, axis=0)
	#print image.shape
    image = np.expand_dims(image, axis=3)
	#print image.shape
	# load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])
   
	# classify the input image
    lists = model.predict(image)[0]
    print(" ")
    print("  Devanagari character recognition")
    print("___________________________________")
    print("")
    print("")
    print ("The letter is ",lab[np.argmax(lists)])
    print("")
    print("")
    label =labels[np.argmax(lists)]
     # draw the label on the image
    output = imutils.resize(orig, width=400)
    
  
    cv2.putText(output,label, (50, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	   0.9, (255, 255, 0), 2)
    output
  
  

    # write the output image
    cv2.imwrite("output.jpg",output)


    
