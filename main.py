import shutil
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
from imutils import resize
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array


def make_image_pred_ready(image_path, width, height):
    images = []
    image = cv2.imread(image_path)
    (img_height, img_width) = image.shape[:2]
    d_width = 0
    d_height = 0
    if img_width < img_height:
        image = resize(image, width=width, inter=cv2.INTER_AREA)
        d_height = int((image.shape[0] - height) / 2.0)
    else:
        image = resize(image, height=height, inter=cv2.INTER_AREA)
        d_width = int((image.shape[1] - width) / 2.0)

    (res_height, res_width) = image.shape[:2]
    image = image[d_height:res_height - d_height, d_width:res_width - d_width]
    image = img_to_array(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    images.append(image)
    return np.array(images)


def get_color_code(category_index):
    color = "#4db6ac"
    if category_index == 4 or category_index == 6:
        color = "#ee6e73"
    return color


def get_max_prob(probs):
    prob_list = probs[0]
    max_val = max(prob_list)
    percent = (max_val * 100)
    percent_str = str(format(percent, '.2f'))
    percent_parts = percent_str.split('.')
    return percent_parts


app = FastAPI()
templates = Jinja2Templates(directory="static")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def handle_upload(file: UploadFile = File(...)):
    destination_path = "./uploads/" + file.filename
    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    img_data = make_image_pred_ready(destination_path, 224, 224)
    img_fl_norm = img_data.astype("float") / 255.0
    recycling_model = load_model('recycling_resnet_152.h5')
    pred = recycling_model.predict(img_fl_norm, batch_size=1)
    parts = get_max_prob(pred)
    category_index = pred.argmax(axis=1)[0]
    label_names = ["Cardboard", "Egg carton", "Glass bottle", "Metal", "Plastic Bag", "Rigid Plastic", "Takeaway Cup"]
    return {"label": label_names[category_index],
            "contamination": get_color_code(category_index),
            "integer_part": parts[0],
            "fraction_part": parts[1]
            }
