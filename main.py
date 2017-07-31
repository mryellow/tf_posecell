import math
import numpy as np
import tensorflow as tf

import pyglet
import random
from pyglet.gl import *
from posevis import Window

sess = tf.Session()

dim      = 3
dim_mid  = dim / 2;
shape    = [dim, dim, dim]
in_shape = (3,)

PC_GLOBAL_INHIB = 0.00002
PC_VT_INJECT_ENERGY = 0.15
PC_VT_RESTORE = 0.05
VT_ACTIVE_DECAY = 1.0

def norm2d(var, x, y, z):
    return 1.0 / (var * math.sqrt(2.0 * math.pi)) \
        * math.exp(( \
            - (x - dim_mid) * (x - dim_mid) \
            - (y - dim_mid) * (y - dim_mid) \
            - (z - dim_mid) * (z - dim_mid) \
        ) / (2.0 * var * var))

def gen_pdf(var):
    total = 0
    res = []

    for k in xrange(dim):
        for j in xrange(dim):
            for i in xrange(dim):
                norm = norm2d(var, i, j, k)
                res.append(norm)
                total += norm

    for x in xrange(len(res)):
        res[x] /= total

    return res

# Weights for each PoseCell based on PDF
def gen_weights(pdf):
    res = []
    for k in xrange(dim):
        res.append([])
        for j in xrange(dim):
            res[k].append([])
            for i in xrange(dim):
                res[k][j].append(gen_weight(pdf, k, j, i))

    return np.asarray(res)

# Roll the PDF until aligned with this cell
def gen_weight(pdf, x, y, z):
    res = np.roll(pdf, ((z-dim_mid) * dim * dim) + ((y-dim_mid) * dim) + ((z-dim_mid) + 1))
    res = np.reshape(res, shape)
    return res


posecells = tf.Variable(tf.ones(shape, tf.float64), name="PoseCells")

# Initalise globals
tf.global_variables_initializer().run(session=sess)

# TODO: DEBUG
posecells_updated = tf.scatter_nd_update(posecells, [[0,0,0]], [1])

# Generate self-connection weights
excite_pdf = gen_pdf(1)
inhibi_pdf = gen_pdf(2)

zeros = tf.zeros(shape, tf.float64, name="Zeros")
excite_weights = tf.constant(gen_weights(excite_pdf), name="WeightsExcite")
inhibi_weights = tf.constant(gen_weights(inhibi_pdf), name="WeightsInhibit")
print('Generated: Weights')

# Local Excite
posecells_reshaped = tf.reshape(posecells_updated, [-1, 1, 1, dim, dim, dim])
excite = tf.squeeze(tf.tensordot(posecells_reshaped, excite_weights, axes=dim), name="Excite")
#excite = tf.assign(posecells, tf.add(posecells, tf.squeeze(tf.tensordot(posecells_reshaped, excite_weights, axes=dim))))
res = sess.run(excite)
print('excite', res, res.shape)

# Local Inhibit
# FIXME: Reshape needed? Keep the shape and reshape once?
excite_reshaped = tf.reshape(excite, [-1, 1, 1, dim, dim, dim])
inhibi = tf.subtract(excite, tf.squeeze(tf.tensordot(excite_reshaped, inhibi_weights, axes=dim)), name="Inhibit")
#inhibi = tf.assign(posecells, tf.subtract(posecells, tf.squeeze(tf.tensordot(posecells_reshaped, inhibi_weights, axes=dim))), name="Inhibit")
res = sess.run(inhibi)
print('inhibi', res, res.shape)

# Global Inhibit
global_applied = tf.subtract(inhibi, PC_GLOBAL_INHIB)
global_limited = tf.where(tf.greater_equal(inhibi, PC_GLOBAL_INHIB), x=global_applied, y=zeros, name="GlobalInhibit")
res = sess.run(global_limited)
print('global_limited', res, res.shape)

#test = tf.greater_equal(inhibi, PC_GLOBAL_INHIB)
#res = sess.run(test)
#print('test', res, res.shape)

# Normalisation
norm_posecells = tf.realdiv(global_limited, tf.reduce_sum(global_limited), name="Norm")
res = sess.run(norm_posecells)
print('norm_posecells', res, res.shape)

process = tf.assign(posecells, norm_posecells)
#process = None

def main(_):
    writer = tf.summary.FileWriter("/tmp/tf_posecell_graph", sess.graph)

    res = None

    window = Window()
    window.dim = dim

    def update(dt):
        #print(window.x, window.y, window.z)

        # TODO: Do stuff
        res = sess.run(process, feed_dict={})

        # Re-scale between [0, 1] for rendering (transparency percentage)
        data = res
        #data = []
        print(data)
        max_activation = 0
        min_activation = 1
        for x in xrange(len(data)):
            for y in xrange(len(data[x])):
                for z in xrange(len(data[x][y])):
                    max_activation = max(max_activation, data[x][y][z])
                    min_activation = min(min_activation, data[x][y][z])

        for x in xrange(len(data)):
            for y in xrange(len(data[x])):
                for z in xrange(len(data[x][y])):
                    data[x][y][z] = (data[x][y][z] - min_activation) / (max_activation - min_activation)

        for x in xrange(len(data)):
            for y in xrange(len(data[x])):
                for z in xrange(len(data[x][y])):
                    window.voxel.set(x, y, z, data[x][y][z])

    pyglet.clock.schedule(update)

    pyglet.app.run()
    writer.close()

if __name__ == '__main__':
    tf.app.run(main=main)
