import math
import numpy as np
import tensorflow as tf

import pyglet
import random
from pyglet.gl import *
from posevis import Window

from weights import Weights

sess = tf.Session()

dim      = 7
dim_mid  = dim / 2;
shape    = [dim, dim, dim]
in_shape = (3,)
PC_C_SIZE_TH  = (2.0 * math.pi) / dim;

PC_GLOBAL_INHIB = 0.00002
PC_VT_INJECT_ENERGY = 0.15
PC_VT_RESTORE = 0.05
VT_ACTIVE_DECAY = 1.0
PC_CELL_X_SIZE = 1.0

weights = Weights(dim, dim)

pose_input = tf.placeholder(tf.int64, shape=in_shape, name="PoseInput")
view_input = tf.placeholder(tf.int64, shape=in_shape, name="ViewInput")

view_decay = tf.Variable(tf.fill(shape, VT_ACTIVE_DECAY), name="ViewDecay")
posecells  = tf.Variable(tf.zeros(shape, tf.float32), name="PoseCells")

# Generate self-connection weights
excite_pdf = weights.pdf(1)
inhibi_pdf = weights.pdf(2)

excite_weights = tf.constant(weights.attractor(excite_pdf), dtype=tf.float32, name="WeightsExcite")
inhibi_weights = tf.constant(weights.attractor(inhibi_pdf), dtype=tf.float32, name="WeightsInhibit")

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
# FIXME: Reshape needed? Keep the shape and reshape once? Well just don't `squeeze` `excite` first.
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
# TODO: VT from camera's pointing in different directions. Relative rad.
# `vt_delta_pc_th = vt_rad / (2.0*M_PI) * PC_DIM_TH;`
# `double pc_th_corrected = pcvt->pc_th + vt_rad / (2.0*M_PI) * PC_DIM_TH;`
inject = tf.scatter_nd_add(posecells, [view_input], [energy], name="Inject")
#res = sess.run(inject, feed_dict={
#    view_input: [0, 0, 0]
#})
#print('inject', res, res.shape)

# Slightly restore decay and Limit to VT_ACTIVE_DECAY
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1063
#decay_restore = tf.subtract(decay_bump, PC_VT_RESTORE, name="RestoreDecay")
decay_restore = tf.assign(view_decay, tf.maximum(VT_ACTIVE_DECAY, tf.subtract(decay_bump, PC_VT_RESTORE)), name="RestoreDecay")

######################
## Path Integration ##
######################

rots_idx, rots_val   = weights.rotation()
trans_idx, trans_val = weights.translation()

trans_weights_dense = tf.sparse_to_dense(
    sparse_indices=trans_idx,
    output_shape=[dim, dim, dim, dim, dim, dim],
    sparse_values=trans_val,
    default_value=0,
    validate_indices=False
)
rots_weights_dense = tf.sparse_to_dense(
    sparse_indices=rots_idx,
    output_shape=[dim, dim, dim, dim, dim, dim],
    sparse_values=rots_val,
    default_value=0,
    validate_indices=False
)

posecells_reshaped = tf.reshape(posecells, [-1, 1, 1, dim, dim, dim])
translate = tf.tensordot(posecells_reshaped, trans_weights_dense, axes=3, name="Translate")
rotate    = tf.tensordot(translate, rots_weights_dense, axes=3, name="Rotate")
trans = tf.squeeze(rotate)
#res = sess.run(trans, feed_dict={
#    vtrans: 1.0,
#    vrot: 1.0
#})
#print('trans', res, res.shape)

#################
## "Callbacks" ##
#################

process = tf.assign(posecells, norm_posecells)

on_view_template = tf.assign(posecells, inject, name="OnViewTemplate")

#on_odo = tf.assign(posecells, tf.scatter_nd_add(posecells, [path_section], [path_energy]), name="OnOdo")
#on_odo = tf.assign(posecells, tf.scatter_nd_add(posecells, [pose_input], [1]), name="OnOdo")

on_odo = tf.assign(posecells, trans)

#find_best = tf.arg_max(posecells, dimension=0)
find_best = tf.argmax(posecells, axis=0)

# Inject some energy at the start.
posecell_init = tf.assign(posecells, tf.scatter_nd_add(posecells, [[dim_mid,dim_mid,dim_mid]], [1]), name="InitPoseCell")
posecell_init.eval(session=sess)

def main(_):
    writer = tf.summary.FileWriter("/tmp/tf_posecell_graph", sess.graph)

    res = None

    window = Window()
    # So it knows when to wrap inputs
    window.dim = dim

    # Kickstart PoseCells
    window.pose = [dim_mid, dim_mid, dim_mid]
    window.view = [-99, -99, -99]
    window.pose_last = window.pose[:]
    window.view_last = window.view[:]

    window.timestep = 0

    def update(dt):
        window.timestep += 1

        # Do stuff
        data = None
        tasks = []
        # TODO: Prevent injection into recent view templates
        if window.pose_last != window.pose:
            #print('pose', window.pose)
            print('vel', window.vtrans['x'], window.vrot['z'])
        #    tasks.append(on_odo)
        #if window.view_last != window.view:
        #    print('view', window.view)
        #    tasks.append(on_view_template)
        #    tasks.append(decay_restore)

        if window.view[0] >= 0:
            print('view', window.view)
            tasks.append(on_view_template)
            tasks.append(decay_restore)

        tasks.append(process)
        tasks.append(on_odo)

        #tasks.append(find_best)

        res = sess.run(tasks, feed_dict={
            #pose_input: window.pose,
            #view_input: window.view,
            weights.vtrans: window.vtrans['x'],
            weights.vrot: window.vrot['z']
        })

        #print(res)

        #print('best', res[len(tasks)-1])

        window.pose_last = window.pose[:]
        window.view_last = window.view[:]

        # Re-scale between [0, 1] for rendering (transparency percentage)
        if res and len(res) > 0:
            data = res[0]
        else:
            return

        max_activation = 0
        min_activation = 1
        for x in xrange(len(data)):
            for y in xrange(len(data[x])):
                for z in xrange(len(data[x][y])):
                    if data[x][y][z] > 0:
                        max_activation = max(max_activation, data[x][y][z])
                        min_activation = min(min_activation, data[x][y][z])
                    else:
                        max_activation = max(max_activation, 0.0)
                        min_activation = min(min_activation, 0.0)

        for x in xrange(len(data)):
            for y in xrange(len(data[x])):
                for z in xrange(len(data[x][y])):
                    if max_activation - min_activation > 0:
                        data[x][y][z] = (data[x][y][z] - min_activation) / (max_activation - min_activation)
                    else:
                        data[x][y][z] = 0.0

        for x in xrange(len(data)):
            for y in xrange(len(data[x])):
                for z in xrange(len(data[x][y])):
                    # TODO: Switch these axis around and draw axis in window
                    window.voxel.set(x, y, z, data[x][y][z])

    pyglet.clock.schedule(update)

    pyglet.app.run()
    writer.close()

if __name__ == '__main__':
    tf.app.run(main=main)
