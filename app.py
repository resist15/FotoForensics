from flask import Flask, render_template, request, redirect, url_for,session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from keras.models import load_model
import numpy as np
import os
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from numpy import loadtxt
from PIL import Image, ImageChops, ImageEnhance
import itertools
from sqlalchemy.orm import backref
from io import BytesIO
import re

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'secret-key'
#app.run(debug=True)
model = load_model("model_casia_run1.h5")
# app.config['UPLOAD_FOLDER'] = r'uploads'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    images = db.relationship('Images', backref='user', lazy=True)
    # def __repr__(self):
    #     return f'<User {self.username}>'

class Images(db.Model):
    image_id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    filename = db.Column(db.String(50), nullable=False)
    data = db.Column(db.LargeBinary)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # def __repr__(self):
    #     return f'<Image {self.filename}>'



@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'confirm_password' in request.form:
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        account = User.query.filter_by(username=username).first()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must not contain any special characters!'
        elif not username or not password or not confirm_password:
            msg = 'Please fill out the form !'
        elif password != confirm_password:
            msg = 'Passwords do not match.'
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            msg = 'You have successfully registered!'
            # return render_template('signup.html', msg=msg)
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
        return redirect(url_for('login'))
    return render_template('signup.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        session['logged_in'] = True
        session['username'] = username
        global user
        user = User.query.filter_by(username=username).first()
        if not user or user.password != password:
            error = 'Invalid username or password.'
            return render_template('login.html', error=error)
        #return redirect(url_for('upload'))
        return render_template('home.html')
    return render_template('login.html')


@app.route('/upload', methods=['GET','POST'])
def upload_image():
    msg = ''
    # Check if a file was submitted
    if 'file' not in request.files:
        msg = 'No file submitted'
        return render_template('home.html',user=user,msg=msg)

    # Get the submitted file
    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        msg = 'Empty file submitted'
        return render_template('home.html',user=user,msg=msg)
    file.save('static/' + file.filename)

    # Preprocess the image
    global image
    path = 'static/' + file.filename
    image1= prepare_image(path)
    image = image1.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    class_names = ['Tampered', 'Original']
    y_pred_class = np.argmax(y_pred, axis = 1)[0]
    print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')
    if (class_names[y_pred_class]) == 'Original':
            image = Images(filename=file.filename, data=file.read(),user=user)
            db.session.add(image)
            db.session.commit()
    return render_template('home.html',result = class_names[y_pred_class], images = file.filename)
image_size = (128, 128)

def prepare_image(image):
    # Convert the image to ELA and resize it to the desired size
    ela_image = ela(image, 90)
    resized_ela_image = ela_image.resize(image_size)

    # Flatten the image and normalize its pixel values
    flattened_image = np.array(resized_ela_image).flatten() / 255.0

    return flattened_image
def ela(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    image = Image.open(path).convert('RGB')
    image.save(path, 'JPEG', quality = quality)
    temp_image = Image.open(path)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image
@app.route('/view', methods=['GET','POST'])
def show_image():
    user = User.query.filter_by(username=session['username']).first()
    image = Images.query.filter_by(user_id=user.id)
    return render_template('view_image.html', images=image)

@app.route('/about', methods=['GET','POST'])
def about_us():
    return render_template('aboutus.html')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('index'))    

if __name__ == '__main__':
    db.create_all()
    #app.run(debug=True)
