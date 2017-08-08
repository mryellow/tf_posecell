import math
import numpy as np
import tensorflow as tf

import pyglet
import random
from pyglet.gl import *
from posevis import Window

from weights import Weights

sess = tf.Session()

dim_th = 7
dim_xy = 7
mid_th = dim_th / 2;
mid_xy = dim_xy / 2;
shape  = [dim_th, dim_xy, dim_xy]
in_shape = (3,)

PC_GLOBAL_INHIB = 0.00002
PC_VT_INJECT_ENERGY = 0.15
PC_VT_RESTORE = 0.05
VT_ACTIVE_DECAY = 1.0

weights = Weights(dim_th, dim_xy)

#pose_input = tf.placeholder(tf.int64, shape=in_shape, name="PoseInput")
view_input = tf.placeholder(tf.int64, shape=in_shape, name="ViewInput")

# TODO: VT stores it's own decay.
view_decay = tf.Variable(tf.fill(shape, VT_ACTIVE_DECAY), name="ViewDecay")
posecells  = tf.Variable(tf.zeros(shape, tf.float32), name="PoseCells")

# Generate self-connection weights
excite_weights = tf.constant(weights.attractor(1), dtype=tf.float32, name="WeightsExcite")
inhibi_weights = tf.constant(weights.attractor(2), dtype=tf.float32, name="WeightsInhibit")

# Initalise globals
tf.global_variables_initializer().run(session=sess)

####################
## Excite/Inhibit ##
####################

# TODO: Sparse update from left most corner with smaller PDF "Activity Packet"

# Local Excite
posecells_reshaped = tf.reshape(posecells, [-1, 1, 1, dim_th, dim_xy, dim_xy])
excite = tf.squeeze(tf.tensordot(posecells_reshaped, excite_weights, axes=3, name="Excite"))

# Local Inhibit
# FIXME: Reshape needed? Keep the shape and reshape once? Well just don't `squeeze` `excite` first.
excite_reshaped = tf.reshape(excite, [-1, 1, 1, dim_th, dim_xy, dim_xy])
inhibi = tf.subtract(excite, tf.squeeze(tf.tensordot(excite_reshaped, inhibi_weights, axes=3, name="Inhibit")))

# Global Inhibit
global_limited = tf.maximum(0.0, tf.subtract(inhibi, PC_GLOBAL_INHIB))

# Normalisation
norm_posecells = tf.realdiv(global_limited, tf.reduce_sum(global_limited), name="Norm")

#########################
## Inject and Decay VT ##
#########################

# TODO: Decay linked to VT record

# Add to decay, except for current view
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1043
decay_bump = tf.scatter_nd_add(view_decay, [view_input], [VT_ACTIVE_DECAY], name="AddDecay")

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

# Inject energy given by decay
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1038
# TODO: VT from camera's pointing in different directions. Relative rad.
# `vt_delta_pc_th = vt_rad / (2.0*M_PI) * PC_DIM_TH;`
# `double pc_th_corrected = pcvt->pc_th + vt_rad / (2.0*M_PI) * PC_DIM_TH;`

inject = tf.scatter_nd_add(posecells, [view_input], [energy], name="Inject")

# Slightly restore decay and Limit to VT_ACTIVE_DECAY
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1063
decay_restore = tf.assign(view_decay, tf.maximum(VT_ACTIVE_DECAY, tf.subtract(decay_bump, PC_VT_RESTORE)), name="RestoreDecay")

######################
## Path Integration ##
######################

rots_idx, rots_val   = weights.rotation()
trans_idx, trans_val = weights.translation()

trans_weights_dense = tf.sparse_to_dense(
    sparse_indices=trans_idx,
    output_shape=[dim_th, dim_xy, dim_xy, dim_th, dim_xy, dim_xy],
    sparse_values=trans_val,
    default_value=0,
    validate_indices=False
)
rots_weights_dense = tf.sparse_to_dense(
    sparse_indices=rots_idx,
    output_shape=[dim_th, dim_xy, dim_xy, dim_th, dim_xy, dim_xy],
    sparse_values=rots_val,
    default_value=0,
    validate_indices=False
)

posecells_reshaped = tf.reshape(posecells, [-1, 1, 1, dim_th, dim_xy, dim_xy])
translate = tf.tensordot(posecells_reshaped, trans_weights_dense, axes=3, name="Translate")
rotate    = tf.tensordot(translate, rots_weights_dense, axes=3, name="Rotate")
trans = tf.squeeze(rotate)

#################
## "Callbacks" ##
#################

process = tf.assign(posecells, norm_posecells)

on_view_template = tf.assign(posecells, inject, name="OnViewTemplate")

#on_odo = tf.assign(posecells, tf.scatter_nd_add(posecells, [path_section], [path_energy]), name="OnOdo")
#on_odo = tf.assign(posecells, tf.scatter_nd_add(posecells, [pose_input], [1]), name="OnOdo")

on_odo = tf.assign(posecells, trans)

# TODO: Find full index.
#find_best = tf.argmax(posecells, axis=0)

# Inject some energy at the start.
posecell_init = tf.assign(posecells, tf.scatter_nd_add(posecells, [[mid_th, mid_xy, mid_xy]], [1]), name="InitPoseCell")
posecell_init.eval(session=sess)

def main(_):
    writer = tf.summary.FileWriter("/tmp/tf_posecell_graph", sess.graph)

    res = None

    window = Window(dim_th, dim_xy)

    def update(dt):
        window.timestep += 1

        # Do stuff
        data = None
        tasks = []

        # TODO: Prevent injection into recent view templates
        #print('view', window.view)
        #tasks.append(on_view_template)
        #tasks.append(decay_restore)

        tasks.append(process)
        tasks.append(on_odo)

        #tasks.append(find_best)

        res = sess.run(tasks, feed_dict={
            #pose_input: window.pose,
            #view_input: window.view,
            weights.vtrans: window.vtrans['x'],
            weights.vrot: window.vrot['z']
        })

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
