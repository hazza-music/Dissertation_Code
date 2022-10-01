import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
from keras import optimizers
from keras import metrics
from keras import Model
from keras.applications import resnet

tf.saved_model.load('siamese_1',"C:/Users/Hairybot/.keras/models")