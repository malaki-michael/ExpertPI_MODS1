import cv2
import requests


def process_image(image, host, port, model_name):
    output = requests.post(f"http://{host}:{port}/predictions/{model_name}",
                           data=cv2.imencode(".jpg", image)[1].tostring())
    return output.json()
