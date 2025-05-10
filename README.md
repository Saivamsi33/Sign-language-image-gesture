# Sign-language-image-gesture
This is a sign language project where everything is made from scratch but leveraging the power of transfer learning of YOLOV8 deep learning model to train custom data. Hope you find this project helpful :)
NOTE - This project is only on a small scale data where I have trained it only from numbers 0 - 9. Here is a reference to the sign gestures from 0 - 9. Make sure you are using python 3.9+.
Image reference to try for yourself - https://www.shutterstock.com/image-photo/numbers-110-sign-language-250039771

How to run:
1. Install the requirements.txt file libraries using "pip install -r requirements.txt".
2. Now run the command "uvicorn main:app --host 0.0.0.0 --port 8080" in the git bash terminal OR just run the main.py file.
3. VOILÀ!!!

Frameworks and tools used :
Ultralytics, YOLOv8, CVAT, Google Colab, FastAPI, Python, HTML, CSS.

Making of this project - So I have taken my own custom data of sign language numbers from 0 to 9. 10 classes which have 150 images each. I had to manually annotate and label each and every single image over 1500 images using [CVAT](https://www.cvat.ai/pricing/cvat-online). 1000 images for training and 500 for validation. Taken all the images and trained YOLOv8 model in google colab.
