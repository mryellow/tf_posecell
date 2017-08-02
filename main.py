import math
import numpy as np
import tensorflow as tf

import pyglet
import random
from pyglet.gl import *
from posevis import Window

sess = tf.Session()

dim      = 14
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
                # TODO: Sparsify, indexes and values, over a min limit
                res[k][j].append(gen_weight(pdf, k, j, i))

    return np.asarray(res)

# Roll the PDF until aligned with this cell
def gen_weight(pdf, x, y, z):
    # Fully connected
    res = np.roll(pdf, ((x-dim_mid) * dim * dim) + ((y-dim_mid) * dim) + (z-dim_mid))
    # TODO: Locally connected "Activity Packets"
    # [[left, most, corner], [smaller, p, d, f]]
    res = np.reshape(res, shape)
    return res

pose_input = tf.placeholder(tf.int64, shape=in_shape, name="PoseInput")
view_input = tf.placeholder(tf.int64, shape=in_shape, name="ViewInput")

view_decay = tf.Variable(tf.fill(shape, VT_ACTIVE_DECAY), name="ViewDecay")
posecells  = tf.Variable(tf.ones(shape, tf.float32), name="PoseCells")

# Generate self-connection weights
excite_pdf = gen_pdf(1)
inhibi_pdf = gen_pdf(2)

excite_weights = tf.constant(gen_weights(excite_pdf), dtype=tf.float32, name="WeightsExcite")
inhibi_weights = tf.constant(gen_weights(inhibi_pdf), dtype=tf.float32, name="WeightsInhibit")

# Initalise globals
tf.global_variables_initializer().run(session=sess)

# TODO: DEBUG
#posecells_updated = tf.scatter_nd_update(posecells, [[dim_mid,dim_mid,dim_mid]], [2])

####################
## Excite/Inhibit ##
####################

# TODO: Sparse update from left most corner with smaller PDF "Activity Packet"

# Local Excite
posecells_reshaped = tf.reshape(posecells, [-1, 1, 1, dim, dim, dim])
excite = tf.squeeze(tf.tensordot(posecells_reshaped, excite_weights, axes=3, name="Excite"))
#res = sess.run(excite)
#print('excite', res, res.shape)

# Local Inhibit
# FIXME: Reshape needed? Keep the shape and reshape once?
excite_reshaped = tf.reshape(excite, [-1, 1, 1, dim, dim, dim])
inhibi = tf.subtract(excite, tf.squeeze(tf.tensordot(excite_reshaped, inhibi_weights, axes=3, name="Inhibit")))
#res = sess.run(inhibi)
#print('inhibi', res, res.shape)

# Global Inhibit
global_limited = tf.maximum(0.0, tf.subtract(inhibi, PC_GLOBAL_INHIB))
#res = sess.run(global_limited)
#print('global_limited', res, res.shape)

# Normalisation
norm_posecells = tf.realdiv(global_limited, tf.reduce_sum(global_limited), name="Norm")
#res = sess.run(norm_posecells)
#print('norm_posecells', res, res.shape)

#########################
## Inject and Decay VT ##
#########################

# Add to decay, except for current view
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1043
decay_bump = tf.scatter_nd_add(view_decay, [view_input], [VT_ACTIVE_DECAY], name="AddDecay")
#res = sess.run(decay_bump, feed_dict={
#    view_input: [0, 0, 0]
#})
#print('decay_bump', res, res.shape)

# Calculate energy given decay
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1047
energy = tf.maximum(
    0.0,
    tf.divide(
        PC_VT_INJECT_ENERGY * 1.0,
        tf.multiply(
            30.0,
            tf.subtract(
                30.0,
                tf.exp(
                    tf.multiply(
                        1.2,
                        decay_bump[view_input[0],view_input[1],view_input[2]]
                    )
                )
            )
        ), name="CalcEnergy"
    )
)
#res = sess.run(energy, feed_dict={
#    view_input: [0, 0, 0]
#})
#print('energy', res, res.shape)

# Inject energy given by decay
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1038
inject = tf.scatter_nd_add(posecells, [view_input], [energy], name="Inject")

# Slightly restore decay and Limit to VT_ACTIVE_DECAY
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1063
#decay_restore = tf.subtract(decay_bump, PC_VT_RESTORE, name="RestoreDecay")
decay_restore = tf.assign(view_decay, tf.maximum(VT_ACTIVE_DECAY, tf.subtract(decay_bump, PC_VT_RESTORE)), name="RestoreDecay")

######################
## Path Integration ##
######################

# % work out the weight contribution to the NE cell from the SW, NW, SE cells
#take_from =

# multiple by the contributing weight
#give_to =

#################
## "Callbacks" ##
#################

process = tf.assign(posecells, norm_posecells)

on_view_template = tf.assign(posecells, inject, name="OnViewTemplate")

#on_odo = tf.assign(posecells, tf.scatter_nd_add(posecells, [path_section], [path_energy]), name="OnOdo")
#on_odo = tf.assign(posecells, tf.scatter_nd_add(posecells, [pose_input], [1]), name="OnOdo")

# Inject some energy at the start. (Why is this needed?)
one_time = tf.assign(posecells, tf.scatter_nd_add(posecells, [dim_mid,dim_mid,dim_mid], [1]), name="OneTime")

def main(_):
    writer = tf.summary.FileWriter("/tmp/tf_posecell_graph", sess.graph)

    res = None

    window = Window()
    # So it knows when to wrap inputs
    window.dim = dim

    # Kickstart PoseCells
    window.pose = [dim_mid, dim_mid, dim_mid]
    window.view = [dim_mid, dim_mid, dim_mid]
    window.pose_last = window.pose[:]
    window.view_last = window.view[:]
    sess.run(one_time, feed_dict={
        pose_input: window.pose
    })

    def update(dt):

        # Do stuff
        data = None
        tasks = []
        # TODO: Prevent injection into recent view templates

        if window.pose_last != window.pose:
            print('pose', window.pose)
            tasks.append(on_odo)
        if window.view_last != window.view:
            print('view', window.view)

        tasks.append(on_view_template)
        tasks.append(decay_restore)

        tasks.append(process)

        res = sess.run(tasks, feed_dict={
            pose_input: window.pose,
            view_input: window.view
        })

        window.pose_last = window.pose[:]
        window.view_last = window.view[:]

        # Re-scale between [0, 1] for rendering (transparency percentage)
        if res and len(res) > 0:
            data = res[0]
            #print(data)
        else:
            return

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
