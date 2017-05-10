import tensorflow as tf
import numpy as np
from model import Autoencoder
import sys
import os

# Get the features
np_data = np.load(sys.argv[1])
n = len(np_data)

slices = []

# Preprocess the features
for k in range(n):
    tmp = np_data[k]
    start = range(0, len(tmp) + 1, 50)
    stop = start[2:]
    L = [tmp[i : j] for i, j in zip(start, stop)]
    slices += L

slices = np.array(slices)
n = len(slices)

# create autoencoder
ae = Autoencoder()
ae.build_model()
ae.train()

# Add embeddings
projs = tf.Variable(tf.truncated_normal([n, ae.bottleneck_dim]), name="projections")
projs_ass = tf.assign(projs, ae.encoder)
proj_saver = tf.train.Saver([projs])

# Do the training
init = tf.global_variables_initializer()
sess = tf.Session()

# File writer for tensorboard
if "board" in sys.argv:
    os.system("rm -rf /tmp/tf/")
    os.system("killall tensorboard")
    os.system("tensorboard --logdir /tmp/tf/ --port 6006 &")

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("/tmp/tf/train/", sess.graph)

sess.run(init)
print()
print()
print("------------------------------")

train_err = []
val_err = []

iters = 20
if len(sys.argv) > 2:
    iters = int(sys.argv[2])

if len(sys.argv) > 3:
    batch_size = int(sys.argv[3])

for i in range(iters): 
    idx = np.random.permutation(n)
    patterns = slices[idx]

    for j in range(int(np.floor(n/batch_size))):
        pattern = patterns[j * batch_size : (j + 1) * batch_size] 
        sess.run("train_step", feed_dict={"raw_data:0": pattern})

    if i%10 == 0 and "log" in sys.argv:
        m, tr_err, _ = sess.run([merged, "err:0", projs_ass], feed_dict={"raw_data:0": slices})
        train_writer.add_summary(m, i)
        ae.saver.save(sess, "/tmp/tf/model.cpkt", global_step=i)
        proj_saver.save(sess, "/tmp/tf/proj.cpkt", global_step=i)

        # Save in numpy format
        train_err.append(tr_err)
        np.save("logs/train_err", train_err)
    print(i)

# Save trained model
save_path = ae.save(sess, iters)
print("Saved model in {0}".format(save_path))

