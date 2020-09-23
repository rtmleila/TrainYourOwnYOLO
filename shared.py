import json
from io import BytesIO

import boto3
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# constants
REGION = 'us-east-2'
BUCKET = 'stark-rta-augur'


def make_s3_resource():
    return boto3.resource('s3', region_name=REGION)


def create_s3_bucket_obj(s3):
    return s3.Bucket(BUCKET)


def read_image_from_s3(s3_bucket, image_id: str) -> np.array:
    """Load image file from s3.

    Parameters
    ----------
    image_id: string

    Returns
    -------
    np array
        Image array
    """
    if 'png' in image_id:
        key = 'data/' + image_id
    else:
        key = 'data/' + image_id + '.png'
    obj = s3_bucket.Object(key)
    response = obj.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
    return np.array(im)


def load_json_file(s3_bucket, key):
    obj = s3_bucket.Object(key)
    response = obj.get()
    json_data_str = response['Body'].read().decode('utf-8')
    return json.loads(json_data_str)    


def list_image_ids(s3_bucket):
    return [obj.key.replace('data/', '') for obj in s3_bucket.objects.filter(Prefix='data/')]


def plot_image_arr(image: np.array, bw: bool=False):
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize = (20,40))
    cmap = 'Greys' if bw else None
    ax.imshow(image, cmap=cmap)
    return fig, ax
    
    
def plot_bounding_box(image_arr: np.array, bounding_box: tuple):
    # Create a Rectangle patch
    
    fig, ax = plt.subplots(1, figsize = (20,40))
    ax.imshow(image_arr)
    
    x, y, width, height = bounding_box
    rect = patches.Rectangle((x, y),  width, height, fill=False)
    ax.add_patch(rect)    
    return ax