import os, math, time, pickle, gc, cv2
import pandas as pd
import numpy as np
import seaborn as sns
import PIL.ImageOps as ImageOps
import matplotlib.pyplot as plt
from PIL import Image
from utils import *
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD,Adam
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.utils.generic_utils import CustomObjectScope

pd.set_option('max_columns', 100)
sns.set_style("whitegrid")

# Confirm GPU availbility
print("GPU availbility:", K.tensorflow_backend._get_available_gpus())

# Global paths
IMG_PATH = "<your_own_img_path>"
ROOT = "<your_own_root_path>"
MODEL_PATH = ROOT + "4_models\\"
EXPORT_PATH = ROOT + "0_data_lan\\export\\"
DATASET_PATH = ROOT + "0_data_lan\\datasets\\"
RAW_PATH = DATASET_PATH + "raw\\"
MST_PATH = DATASET_PATH + "mst\\"
FACIAL_PATH = RAW_PATH + "facial_landmark\\"
STITCH_EYE_PATH = MST_PATH + "stitched_eyes\\"
MODEL_SAVE_PATH = MODEL_PATH + "0_saved_models\\"
MODEL_VIZ_PATH = MODEL_PATH + "1_train_process\\"
MODEL_PERFORM_PATH = MODEL_PATH + "2_model_performance\\"
PICKLE_PATH = MST_PATH + "stitched_eyes_pickle\\"

# Helper functions for build files
def get_facial_landmarks(df, idx):
    xs = [df.loc[idx, 'x_' + str(i)] for i in range(68)]
    ys = [df.loc[idx, 'y_' + str(i)] for i in range(68)]
    return xs, ys

def get_eye_landmarks(df, idx):
    xs = [df.loc[idx, 'x_' + str(i)] for i in range(36,48)]
    ys = [df.loc[idx, 'y_' + str(i)] for i in range(36,48)]
    return xs, ys

def enlarge_box(left_crop, right_crop, x_enlarge_factor = 0.2, y_enlarge_factor = 0.3):
    (left_x_0, left_y_0, left_x_1, left_y_1), (right_x_0, right_y_0, right_x_1, right_y_1) = left_crop, right_crop
    left_x_enlarge = (left_x_1 - left_x_0) * x_enlarge_factor
    left_y_enlarge = (left_y_1 - left_y_0) * y_enlarge_factor
    left_x_0 -= left_x_enlarge
    left_x_1 += left_x_enlarge
    left_y_0 -= left_y_enlarge
    left_y_1 += left_y_enlarge
    
    right_x_enlarge = (right_x_1 - right_x_0) * x_enlarge_factor
    right_y_enlarge = (right_y_1 - right_y_0) * y_enlarge_factor
    right_x_0 -= right_x_enlarge
    right_x_1 += right_x_enlarge
    right_y_0 -= right_y_enlarge
    right_y_1 += right_y_enlarge
    return (left_x_0, left_y_0, left_x_1, left_y_1), (right_x_0, right_y_0, right_x_1, right_y_1)

def simple_eye_box(df, idx, enlarge = True):
    # Calculates the x,y upper left corner of LEFT eye and the width and height of the box
    left_x_0 = df.loc[idx, 'x_36']
    left_y_0 = min(df.loc[idx, 'y_36'], df.loc[idx, 'y_37'], df.loc[idx, 'y_38'], df.loc[idx, 'y_39'])
    left_x_1 = df.loc[idx, 'x_39']
    left_y_1 = max(df.loc[idx, 'y_36'], df.loc[idx, 'y_41'], df.loc[idx, 'y_40'], df.loc[idx, 'y_39'])
    
    # Calculates the x,y upper left corner of RIGHT eye and the width and height of the box
    right_x_0 = df.loc[idx, 'x_42']
    right_y_0 = min(df.loc[idx, 'y_42'], df.loc[idx, 'y_43'], df.loc[idx, 'y_44'], df.loc[idx, 'y_45'])
    right_x_1 = df.loc[idx, 'x_45']
    right_y_1 = max(df.loc[idx, 'y_42'], df.loc[idx, 'y_47'], df.loc[idx, 'y_46'], df.loc[idx, 'y_45'])
    
    if enlarge:
        return enlarge_box((left_x_0, left_y_0, left_x_1, left_y_1), (right_x_0, right_y_0, right_x_1, right_y_1))
    return (left_x_0, left_y_0, left_x_1, left_y_1), (right_x_0, right_y_0, right_x_1, right_y_1)

def crop_resize_stitch(img, left_crop, right_crop, x_resize, y_resize):
    # Note Image.ANTIALIAS is a high-quality downsampling filter
    img_left = img.crop(left_crop).resize((x_resize, y_resize), Image.ANTIALIAS)
    img_right = img.crop(right_crop).resize((x_resize, y_resize), Image.ANTIALIAS)
    stitched = Image.new("RGBA", (x_resize * 2, y_resize))
    x = 0
    for i in [img_left, img_right]:
        stitched.paste(i, (x, 0))
        x += i.size[0]
    return stitched

def calculate_rotation(df, idx):
    # 36 is the lefe most point of the left eye while 45 is the right most point of the right eye
    delta_y = df.loc[idx, 'y_36'] - df.loc[idx, 'y_45']
    delta_x = df.loc[idx, 'x_36'] - df.loc[idx, 'x_45']
    # Return the arctan of the differences in y and x
    return math.degrees(math.atan(delta_y / delta_x))

def rotate_points(xs, ys, cx, cy, alpha):
    ''' xs, ys: set of xs and ys to be modified
        cx, cy: the center of the image
        alpha: the rotation to be applied'''
    x_new, y_new = [], []
    for x,y in zip(xs,yx):
        delta_x, delta_y = x - cx, cy - y1
        length = (delta_x + delta_y) ** 2
        angel = math.atan(delta_y / delta_x) + alpha
        x_new.append(cx + length*math.cos(angel))
        y_new.append(cy + length*math.sin(angel))
    return x_new, y_new

def get_groud_truth(df, idx):
    X = np.mean([x for x in [df.loc[idx, 'tobiiLeftScreenGazeX'], df.loc[idx, 'tobiiRightScreenGazeX']] if x != -1])
    Y = np.mean([x for x in [df.loc[idx, 'tobiiLeftScreenGazeY'], df.loc[idx, 'tobiiRightScreenGazeY']] if x != -1])
    return X,Y

def image_loader(df):
    x_batch = []
    y_batch_coord_x = []
    y_batch_coord_y = []
    df.index = range(len(df))
    for ind in range(len(df)):
        if ind % 10000 == 0: print("Step:", ind)
        read_path = df.loc[ind, 'path'].replace("david_thesis", "david_xu_thesis")
        # Resize and convert
        img = Image.open(read_path).resize((160, 32), Image.ANTIALIAS)
        img = np.array(img)
        # Histogram equalize the images
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img[:,:,0] = cv2.equalizeHist(img[:,:,0])
        img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
        img = preprocess_input(img)
        x_batch.append(img)
        y_batch_coord_x.append(df.loc[ind, 'coord_x']) 
        y_batch_coord_y.append(df.loc[ind, 'coord_y']) 
    
    x_batch = np.array(x_batch, dtype = np.float32)
    y_batch_coord_x = np.array(y_batch_coord_x, dtype = np.float32)
    y_batch_coord_y = np.array(y_batch_coord_y, dtype = np.float32)
    return x_batch, y_batch_coord_x, y_batch_coord_y

def get_facial_landmarks(df, idx):
    xs = [df.loc[idx, 'x_' + str(i)] for i in range(68)]
    ys = [df.loc[idx, 'y_' + str(i)] for i in range(68)]
    return xs, ys

def get_eye_landmarks(df, idx):
    xs = [df.loc[idx, 'x_' + str(i)] for i in range(36,48)]
    ys = [df.loc[idx, 'y_' + str(i)] for i in range(36,48)]
    return xs, ys

def enlarge_box(left_crop, right_crop, x_enlarge_factor = 0.2, y_enlarge_factor = 0.3):
    ''' Enlarge the bounding box around the eye by a factor of 1.2 '''
    (left_x_0, left_y_0, left_x_1, left_y_1), (right_x_0, right_y_0, right_x_1, right_y_1) = left_crop, right_crop
    left_x_enlarge = (left_x_1 - left_x_0) * x_enlarge_factor
    left_y_enlarge = (left_y_1 - left_y_0) * y_enlarge_factor
    left_x_0 -= left_x_enlarge
    left_x_1 += left_x_enlarge
    left_y_0 -= left_y_enlarge
    left_y_1 += left_y_enlarge
    
    right_x_enlarge = (right_x_1 - right_x_0) * x_enlarge_factor
    right_y_enlarge = (right_y_1 - right_y_0) * y_enlarge_factor
    right_x_0 -= right_x_enlarge
    right_x_1 += right_x_enlarge
    right_y_0 -= right_y_enlarge
    right_y_1 += right_y_enlarge
    return (left_x_0, left_y_0, left_x_1, left_y_1), (right_x_0, right_y_0, right_x_1, right_y_1)

def simple_eye_box(df, idx, enlarge = True):
    # Calculates the x,y upper left corner of LEFT eye and the width and height of the box
    left_x_0 = df.loc[idx, 'x_36']
    left_y_0 = min(df.loc[idx, 'y_36'], df.loc[idx, 'y_37'], df.loc[idx, 'y_38'], df.loc[idx, 'y_39'])
    left_x_1 = df.loc[idx, 'x_39']
    left_y_1 = max(df.loc[idx, 'y_36'], df.loc[idx, 'y_41'], df.loc[idx, 'y_40'], df.loc[idx, 'y_39'])
    
    # Calculates the x,y upper left corner of RIGHT eye and the width and height of the box
    right_x_0 = df.loc[idx, 'x_42']
    right_y_0 = min(df.loc[idx, 'y_42'], df.loc[idx, 'y_43'], df.loc[idx, 'y_44'], df.loc[idx, 'y_45'])
    right_x_1 = df.loc[idx, 'x_45']
    right_y_1 = max(df.loc[idx, 'y_42'], df.loc[idx, 'y_47'], df.loc[idx, 'y_46'], df.loc[idx, 'y_45'])
    
    if enlarge:
        return enlarge_box((left_x_0, left_y_0, left_x_1, left_y_1), (right_x_0, right_y_0, right_x_1, right_y_1))
    return (left_x_0, left_y_0, left_x_1, left_y_1), (right_x_0, right_y_0, right_x_1, right_y_1)

def crop_resize_stitch(img, left_crop, right_crop, x_resize, y_resize):
    # Note Image.ANTIALIAS is a high-quality downsampling filter
    img_left = img.crop(left_crop).resize((x_resize, y_resize), Image.ANTIALIAS)
    img_right = img.crop(right_crop).resize((x_resize, y_resize), Image.ANTIALIAS)
    stitched = Image.new("RGBA", (x_resize * 2, y_resize))
    x = 0
    for i in [img_left, img_right]:
        stitched.paste(i, (x, 0))
        x += i.size[0]
    return stitched

# Helper functions for analysis code
def pickle_loader(pickle_names):
    '''pickle_names is the list of pickle names you want to load'''
    for n in pickle_names:
        print("Loading:", n)
        with open(PICKLE_PATH+n+'.pickle', 'rb') as f:
            temp = pickle.load(f)
            temp = np.array(temp, dtype = np.float16)
        if pickle_names.index(n) == 0:
            data = temp
        else:
            data = np.concatenate([data, temp], 0)
        del temp
        gc.collect()
    return data

def histogram_equalize(img):
    '''Histogram equalize the images'''
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    return img

def batch_generator(df, batch_size):
    """This generator use a pandas DataFrame to read images from disk.
    """
    N = df.shape[0]
    while True:
        for start in range(0, N, batch_size):
            start_time = time.time()
            x_batch = []
            y_batch_coord_x = []
            y_batch_coord_y = []
            if start + batch_size > N: break
            for ind in range(start, start + batch_size):
                read_path = df.loc[ind, 'path'].replace("david_thesis", "david_xu_thesis")
                # Resize and convert
                img = Image.open(read_path).resize((160, 32), Image.ANTIALIAS)
                img = np.array(img)
                # Histogram equalize the images
                img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                img[:,:,0] = cv2.equalizeHist(img[:,:,0])
                img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
                img = preprocess_input(img)
                x_batch.append(img)
                y_batch_coord_x.append(df.loc[ind, 'coord_x']) 
                y_batch_coord_y.append(df.loc[ind, 'coord_y'])  
            
            # print("Load time:", time.time() - start_time)
            x_batch = np.array(x_batch, dtype = np.float32)
            y_batch_coord_x = np.array(y_batch_coord_x, dtype = np.float32)
            y_batch_coord_y = np.array(y_batch_coord_y, dtype = np.float32)
            
            yield (x_batch, [y_batch_coord_x, y_batch_coord_y])
   

def write_model_specs(model_name, textList):
    outF = open(MODEL_SAVE_PATH + model_name + ".txt", "w")
    for line in textList:
      outF.write(line)
      outF.write("\n")
    outF.close()

def write_model_performance(model_name, textList):
    df = pd.DataFrame()
    df['model_name'] = [model_name]
    for t in textList:
        spec, value = t.split(":")
        spec = spec.lower()
        df[spec] = [value]
    df.to_csv(MODEL_PERFORM_PATH + model_name + "//model_performance.csv", index = None)