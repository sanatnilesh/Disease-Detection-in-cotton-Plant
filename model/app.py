from flask import Flask, jsonify, request
from IPython.display import display,Image
from scipy import stats as st
from keras.models import model_from_json, load_model
from numpy import array
import numpy as np
import cv2
import skimage.io
import skimage.feature
import scipy
import json

import flask
app = Flask(__name__)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict/<string:imgname>', methods=['POST','GET'])
def predict(imgname):
    feature_row = []
    img=cv2.imread('./upload/{0}.jpg'.format(imgname))
    dst=cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    im=skimage.img_as_ubyte(gray) 
    
    mean=np.mean(im)
    var=np.var(im)
    std=np.std(im)
    
    im=im/32
    x=im.astype(int)
    
    g = skimage.feature.greycomatrix(x,[1],[0],levels=8,symmetric=False,normed=True)
    contrast = skimage.feature.greycoprops(g,'contrast')[0][0]
    feature_row.append(contrast)
    energy = skimage.feature.greycoprops(g,'energy')[0][0]
    feature_row.append(energy)
    homogeneity = skimage.feature.greycoprops(g,'homogeneity')[0][0]
    feature_row.append(homogeneity)
    correlation = skimage.feature.greycoprops(g,'correlation')[0][0]
    feature_row.append(correlation)
    entropy = skimage.measure.shannon_entropy(gray)#base=2 or 10 or np.e.
    feature_row.append(entropy)
    mean_or_avg = mean/32
    feature_row.append(mean_or_avg)
    variance = var/32
    feature_row.append(variance)
    std_dev = std/32
    feature_row.append(std_dev)

    X = array([feature_row])
    Y = model.predict_classes(X)
    Z = model.predict_proba(X)

    index_max = np.argmax(Z[0])
    ans = index_max

    if(ans == 0):
    	str = 'Alternaria'
    	control = 'Seed treatment with Pseudomonas fluorescens and spraying of 0.2%'
    elif(ans == 3):
    	str = 'Bacterial blight'
    	control = 'Seed treatment with authorized antibiotics with cupravit 0.2%'
    elif(ans == 1):
    	str = 'Anthracnose'
    	control = 'Treatment of seed with fungicides like captan, carboxin or thiram at 2g/kg'
    elif(ans == 2):
    	str = 'Cercospora'
    	control = 'Spary formulations containing mancozeb or copper oxychloride at 2kg/ha at initiation of disease'
    elif(str == 4):
    	str = 'Healthy cotton'
    	control = 'No problem you good to go :)'
    else:
    	str = 'Sorry! could not identify'
    	control = 'Oops!'
    
    return json.dumps({'input-image':imgname,'classified-image':str,'Control measures': control})

if __name__ == '__main__':
    app.run(host='192.168.43.113', port=5000)
