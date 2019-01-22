import os
import sys
import glob
import argparse

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing import image

from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def hello():
	files = get_files('1.jpg')
	cls_list = ['Cat', 'Dog']
	net = load_model('model-resnet50-final.h5')
	for f in files:
		img = image.load_img(f,target_size=(224,224))
		if img is None:
			continue
		x = image.img_to_array(img)
		x = preprocess_input(x)
		x = np.expand_dims(x, axis=0)
		pred = net.predict(x)[0]
		top_inds = pred.argsort()[::-1][:5]
		for i in top_inds:
			if pred[i] > 0.75:
				print(cls_list[i])
				html = "<html><body><h1>"
				html += cls_list[i]+"</h1></body></html>"
				return html
			else:
				return "I am Not sure"
    

def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    files = [f for f in files if f.endswith('JPG') or f.endswith('jpg')]

    if not len(files):
        sys.exit('No images found by the given path!')

    return files

if __name__ == '__main__':
	app.run(debug = True)