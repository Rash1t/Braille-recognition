import os
from flask import render_template, Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import datetime


UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        return render_template('data.html', filename=filename)
    return render_template('index.html')


@app.route('/data')
def data():
    return render_template('data.html')


if __name__ == "__main__":
    app.run(debug=True)