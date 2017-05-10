import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt
from model import Autoencoder
from conv1 import ConvEncoder
from conv2 import ConvEncoder2
from conv3 import ConvEncoder3
from vae import VAE
from convskip import ConvSkip
from fullskip import FullSkip
import sys

np_data = np.load(sys.argv[1])

print(np_data)

# create autoencoder
ae = FullSkip()
ae.build_model()
ae.train()

iters = 700
if len(sys.argv) > 1:
    iters = sys.argv[2]

# Restore
sess = tf.Session()
ae.load(sess, iters)
print()
print()
print("------------------------------")

# Test the model
idx = [0, 1, 2, 3, 4, -1, -2, -3, -4]
images = np_data["patterns"][idx]
out = sess.run("raw_out:0", feed_dict={"raw_data:0": images})

# Plot some samples

plt.figure()

for i in range(out.shape[0]):
    #pueh = (out[i] - np.min(out[i])) / (np.max(out[i]) - np.min(out[i]) + 1E-6)
    plt.subplot(3, out.shape[0], i + 1)
    plt.imshow(out[i])
    plt.subplot(3, out.shape[0], i + out.shape[0] + 1)
    plt.imshow(np_data["patterns"][idx[i]]/255)
    plt.subplot(3, out.shape[0], i + out.shape[0]*2 + 1)
    plt.imshow(np_data["targets"][idx[i]]/255)
    #print("Scaled with value: ", np.max(out[i]) - np.min(out[i]) + 1E-6)

plt.show()
