import cv2
import tensorflow as tf
import pytesseract
from PIL import Image
import re
#from readtext.readtext import get_serial_number

CATEGORIES = ["Front", "Back"]

#directory = os.getcwd()

def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("CNN.model")
image = "a.jpeg"

arr = prepare(image)
prediction = model.predict([arr])
prediction = list(prediction[0])
print(CATEGORIES[prediction.index(max(prediction))])
