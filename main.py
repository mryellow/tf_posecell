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


view_input = tf.placeholder(tf.int64, shape=in_shape, name="ViewInput")
view_decay = tf.Variable(tf.fill(shape, VT_ACTIVE_DECAY), name="ViewDecay")
posecells  = tf.Variable(tf.ones(shape, tf.float32), name="PoseCells")

# Generate self-connection weights
excite_pdf = gen_pdf(1)
inhibi_pdf = gen_pdf(2)

const_decay    = tf.fill(shape, VT_ACTIVE_DECAY, name="ConstGlobalDecay")
const_zeros    = tf.zeros(shape, tf.float32, name="Zeros")
excite_weights = tf.constant(gen_weights(excite_pdf), dtype=tf.float32, name="WeightsExcite")
inhibi_weights = tf.constant(gen_weights(inhibi_pdf), dtype=tf.float32, name="WeightsInhibit")

# Initalise globals
tf.global_variables_initializer().run(session=sess)

# TODO: DEBUG
#posecells_updated = tf.scatter_nd_update(posecells, [[0,0,0]], [1])

####################
## Excite/Inhibit ##
####################

# Local Excite
posecells_reshaped = tf.reshape(posecells, [-1, 1, 1, dim, dim, dim])
excite = tf.squeeze(tf.tensordot(posecells_reshaped, excite_weights, axes=dim), name="Excite")
#res = sess.run(excite)
#print('excite', res, res.shape)

# Local Inhibit
# FIXME: Reshape needed? Keep the shape and reshape once?
excite_reshaped = tf.reshape(excite, [-1, 1, 1, dim, dim, dim])
inhibi = tf.subtract(excite, tf.squeeze(tf.tensordot(excite_reshaped, inhibi_weights, axes=dim)), name="Inhibit")
#res = sess.run(inhibi)
#print('inhibi', res, res.shape)

# Global Inhibit
global_applied = tf.subtract(inhibi, PC_GLOBAL_INHIB)
global_limited = tf.where(tf.greater_equal(inhibi, PC_GLOBAL_INHIB), x=global_applied, y=const_zeros, name="GlobalInhibit")
#res = sess.run(global_limited)
#print('global_limited', res, res.shape)

# Normalisation
norm_posecells = tf.realdiv(global_limited, tf.reduce_sum(global_limited), name="Norm")
#res = sess.run(norm_posecells)
#print('norm_posecells', res, res.shape)

process = tf.assign(posecells, norm_posecells)
#process = None

###########
## Decay ##
###########

# Add to decay, except for current view
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1043
decay_bump = tf.scatter_nd_add(view_decay, [view_input], VT_ACTIVE_DECAY, name="DecayBump")

# Calculate energy given decay
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1047
energy = tf.divide(PC_VT_INJECT_ENERGY * 1.0, tf.multiply(30.0, tf.subtract(30.0, tf.exp(tf.multiply(1.2, decay_bump)))), name="CalcEnergy")

# Inject energy given by decay, across all indexes
# TODO: Prevent injection into recent view templates
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1038
inject = tf.add(posecells, energy, name="Inject")

# Slightly restore decay
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1063
decay_restore = tf.subtract(inject, PC_VT_RESTORE, name="DecaySlightRestore")

# Limit decay to VT_ACTIVE_DECAY
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1065
decay_applied = tf.subtract(decay_restore, VT_ACTIVE_DECAY, name="DecayRestore")
decay_limited = tf.where(tf.greater_equal(decay_restore, VT_ACTIVE_DECAY), x=decay_applied, y=const_decay, name="DecayLimit")
on_view_template = tf.assign(view_decay, decay_limited)

# TODO: Path integration `on_odo`

def main(_):
    writer = tf.summary.FileWriter("/tmp/tf_posecell_graph", sess.graph)

    res = None

    window = Window()
    window.dim = dim

    def update(dt):
        #print(window.x, window.y, window.z)

        # TODO: Do stuff
        res = sess.run([process, on_view_template], feed_dict={
            view_input: [window.x, window.y, window.z]
        })

        # Re-scale between [0, 1] for rendering (transparency percentage)
        data = res[0]
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
