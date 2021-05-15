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
    y_h = 60
    x_h = 40
    y = 0    
    while y+y_h<dy:
        x=0
        list_y=[]
        while x+x_h<dx:
            y_b=y
            x_b=x
            x_sp=x_b
            flag=False
            while np.max(img[y_b:y_b+y_h,x_b])==0 and y_b+y_h<dy and x_b+x_h<dx:
                x_b+=1
                if x_b-x_sp>50:
                    flag=True
            while np.max(img[y_b,x_b:x_b+x_h])==0 and y_b+y_h<dy and x_b+x_h<dx:
                y_b+=1
            
            try:
                if len(list_y)>0:
                    if y_b>min(np.array(list_y)-30)-y_h*0.75 and y_b<min(np.array(list_y)-30)-y_h*0.5 :
                        yield ','
                    else:
                        if flag:
                            yield ' '
                        yield img[y_b:y_b+y_h,x_b:x_b+x_h]
                else:
                    if flag:
                        yield ' '
                    yield img[y_b:y_b+y_h,x_b:x_b+x_h]
            except:
                pass
            x=x_b+x_h+15
            list_y.append(y_b+y_h+30)
        y=max(list_y)
        
#Блок для распознавания
def Recognition(image_path):
    text = ""
    #Всякую ерунду вставлять сюда
    im = cv2.imread(image_path)
    im = cv2.resize(im, (2000,2750), interpolation = cv2.INTER_AREA)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp=cv2.filter2D(im,-1,filter)
    sharp=cv2.filter2D(sharp,-1,filter)
    sharp = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV, 5, 4)
    kernel = np.ones((5,5),np.uint8)
    sharp = cv2.dilate(sharp,kernel,iterations = 1)
    sharp = cv2.morphologyEx(sharp, cv2.MORPH_OPEN, kernel)
    sharp = cv2.morphologyEx(sharp, cv2.MORPH_OPEN, kernel)
    sharp = cv2.morphologyEx(sharp, cv2.MORPH_OPEN, kernel)

    for i in range(20,sharp.shape[0]):
        a=np.where(sharp[i,20:int(sharp.shape[1]/2)]==0)
        if a[0].shape[0]>sharp.shape[1]/3:
            x_l=a[0][0]
            y_l=i
            break

    for i in range(40,sharp.shape[0]):
        i=sharp.shape[0]-i
        a=np.where(sharp[i,int(sharp.shape[1]/2):-20]==0)
        if a[0].shape[0]>sharp.shape[1]/3:
            x_r=int(sharp.shape[1]/2)+a[0][-1]
            y_r=i
            break
            
    im=im[y_l+int(im.shape[0]*0.0368): y_r-int(im.shape[0]*0.0368),x_l+int(im.shape[1]*0.024):x_r]

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
        new_x=np.zeros((60,40))
        try:
            m_x=np.array(x[np.where(x[:,None]>0)[0][0]:,np.where(x[None,:]>0)[0][0]:] )
            new_x[:m_x.shape[0],:m_x.shape[1]]+=m_x
            a=skimage.measure.block_reduce(new_x, (20,20), np.max)
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
