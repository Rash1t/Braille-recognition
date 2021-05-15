import os
from flask import render_template, Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import datetime
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import skimage.measure


UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def slice(img):
    dy,dx = img.shape
    y_h = 45
    x_h = 30
    y = 0    
    while y+y_h<dy:
        x=0
        list_y=[]
        while x+x_h<dx:
            y_b=y
            x_b=x
            x_sp=x_b
            flag=False
            flag_next=True
            while np.max(img[y_b:y_b+y_h,x_b])==0 and y_b+y_h<dy and x_b+x_h<dx:
                x_b+=1
                if x_b-x_sp>20:
                    flag=True
                if x_b-x_sp>100:
                    flag_next=False
                    continue
            while np.max(img[y_b,x_b:x_b+x_h])==0 and y_b+y_h<dy and x_b+x_h<dx:
                y_b+=1
                if x_b-x_sp>100:
                    continue
            try:
                if flag_next:
                    if flag:
                        yield ' '
                    yield img[y_b:y_b+y_h,x_b:x_b+x_h]
            except:
                pass
            x=x_b+x_h+15
            list_y.append(y_b+y_h+30)
        y=min(list_y)
        
#Блок для распознавания
def Recognition(image_path):
    text = ""
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    im = cv2.blur(im,(3,3))
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 5, 4)
    im = cv2.medianBlur(im, 3)
    _,im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
    im = cv2.GaussianBlur(im, (3,3), 0)
    _,im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
    
    sliced = list(slice(im))
    
    df1=pd.read_csv('symbol_rus.csv', index_col=0)
    df1.matr=df1.matr.apply(lambda x:ast.literal_eval(x.replace('\n','').replace(' ',',')))
    
    res = []
    for i in range(len(sliced)):
        if type(sliced[i])==str:
            res.append(sliced[i])
            continue
        x=sliced[i]/255
        #x[x>0]=1
        new_x=np.zeros((45,30))
        try:
            m_x=np.array(x[np.where(x[:,None]>0)[0][0]:,np.where(x[None,:]>0)[0][0]:] )
            new_x[:m_x.shape[0],:m_x.shape[1]]+=m_x
            a=skimage.measure.block_reduce(new_x, (15,15), np.max)
            try:
                res.append(df1[df1.matr.apply(lambda x: (x==a).all())].sym.values[0])
            except:
                pass
        except:
            pass
    return text.join(res)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('data.html', filename=filename, text=Recognition("./static/images/" + filename))
    return render_template('index.html')


@app.route('/data')
def data():
    return render_template('data.html')


if __name__ == "__main__":
    app.run()
