import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from keras import backend as K
import tensorflow as tf

def aucMetric(true, pred):

        #We want strictly 1D arrays - cannot have (batch, 1), for instance
    true= K.flatten(true)
    pred = K.flatten(pred)

        #total number of elements in this batch
    totalCount = K.shape(true)[0]

        #sorting the prediction values in descending order
    values, indices = tf.nn.top_k(pred, k = totalCount)   
        #sorting the ground truth values based on the predictions above         
    sortedTrue = K.gather(true, indices)

        #getting the ground negative elements (already sorted above)
    negatives = 1 - sortedTrue

        #the true positive count per threshold
    TPCurve = K.cumsum(sortedTrue)

        #area under the curve
    auc = K.sum(TPCurve * negatives)

       #normalizing the result between 0 and 1
    totalCount = K.cast(totalCount, K.floatx())
    positiveCount = K.sum(true)
    negativeCount = totalCount - positiveCount
    totalArea = positiveCount * negativeCount
    return  auc / totalArea

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_model(model_path, custom_objects={"aucMetric": aucMetric})
        self.new_size = (224, 224)

    def readImage(self, path):
        '''
        Inputs images from a path and converts them to RGB.
        '''
        im = Image.open(path)
        im1 = im.resize(self.new_size)
        im1 = im1.convert("RGB")
        return im1


    def preprocess_image(self, path):
        '''
        Resizes and rescales the images
        '''
        img = np.array(self.readImage(path))
        img = img / 255
        img = img.reshape((1, 224, 224, 3))

        return img

    def predict(self, path):
        img = self.preprocess_image(path)
        return self.model.predict(img)


