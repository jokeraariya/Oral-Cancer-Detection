# Oral Cancer Detection Using Deep Learning

## Overview

Oral Cancer Detection is a deep learning-based web application that analyzes microscopic oral tissue images and predicts whether the sample is cancerous or non-cancerous. The system uses a Convolutional Neural Network (CNN) model developed with TensorFlow and Keras and provides predictions through a Flask-based web interface.

## Features

* Upload microscopic oral tissue images
* Automated image preprocessing
* CNN-based cancer classification
* User-friendly web interface
* Real-time prediction results
* Flask backend integration

## System Architecture

Microscopic Oral Tissue Image

↓

Image Preprocessing

* Resize Image (224×224)
* Normalize Pixel Values
* Convert to Array

↓

CNN Deep Learning Model

* TensorFlow
* Keras
* cancer.h5

↓

Prediction Engine

* Cancer
* Non-Cancer

↓

Flask Backend

↓

Web Interface

## Technology Stack

### Frontend

* HTML
* CSS

### Backend

* Flask
* Python

### Deep Learning

* TensorFlow
* Keras
* NumPy

### Image Processing

* Pillow (PIL)

## Project Structure

```text
Source_code/
│
├── app.py
├── templates/
├── static/
├── data/
├── files/
├── screenshots/
└── .gitignore
```

## Installation

### Clone Repository

```bash
git clone https://github.com/jokeraariya/Oral-Cancer-Detection.git
cd Oral-Cancer-Detection
```

### Install Dependencies

```bash
pip install flask
pip install tensorflow==2.14.0
pip install keras==2.14.0
pip install numpy==1.26.4
pip install pillow
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Screenshots

### System Architecture

Add:

```text
screenshots/architecture.png
```

### Home Page

Add:

```text
screenshots/home_page.png
```

### Upload Page

Add:

```text
screenshots/upload_page.png
```

### Prediction Result

Add:

```text
screenshots/cancer_result.png
```

## Future Enhancements

* Improve model accuracy
* Deploy on cloud platforms
* Add confidence visualization
* Support multiple image formats
* Integrate patient history analysis

## Disclaimer

The trained model file (`cancer.h5`) is not included in this repository due to GitHub file size limitations.

## Author

Aariya

B.Tech Information Technology

Deep Learning | AI/ML | Full Stack Development
