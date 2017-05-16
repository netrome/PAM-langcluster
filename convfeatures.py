import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt
from model import Autoencoder
from conv import ConvEncoder
from deepconv import DeepConvEncoder
import sys
from utils import slice_samples

np_data = np.load(sys.argv[1])
slices = slice_samples(np_data, time_dist=int(sys.argv[3]))
print(slices.shape)

# create autoencoder
#ae = DeepConvEncoder()
ae = Autoencoder(image_dims=[20, 26])
ae.build_model()
ae.train()

iters = 700
if len(sys.argv) > 1:
    iters = sys.argv[2]

# Restore
sess = tf.Session()
ae.load(sess, iters, sys.argv[1])
print()
print()
print("------------------------------")

# Use the model
features = []
for sample in slices:
    out = sess.run("bottleneck:0", feed_dict={"raw_data:0": sample})
    features.append(out)


np.save(sys.argv[1].split(".")[0] + "_convfeatures", np.array(features))


