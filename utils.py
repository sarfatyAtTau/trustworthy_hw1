import gzip
import numpy as np
from os import path
from PIL import Image

def read_hw_data():
    """
    Used to load the data
    """
    curr_dir = path.dirname(path.abspath(__file__))
    fpath = path.join(curr_dir, 'data.npz')
    with gzip.open(fpath, 'rb') as fin:
        x_train = np.load(fin).reshape((-1,28,28))/255.
        y_train = np.load(fin)
        x_val = np.load(fin).reshape((-1,28,28))/255.
        y_val = np.load(fin)
        x_test = np.load(fin).reshape((-1,28,28))/255.
        y_test = np.load(fin)
    return x_train, y_train, x_val, y_val, x_test, y_test

def save_as_im(x, outpath):
    """
    Used to store a numpy array (with values in [0,1] as an image).
    Outpath should specify the extension (e.g., end with ".jpg").
    """
    im = Image.fromarray(x*255.).convert('RGB')
    im.save(outpath)

