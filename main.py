import math
import numpy as np
import tensorflow as tf

import pyglet
import random
from pyglet.gl import *
from posevis import Window

sess = tf.Session()

def norm2d(var, x, y, z, dim_mid):
    return 1.0 / (var * math.sqrt(2.0 * math.pi)) \
        * math.exp(( \
            - (x - dim_mid) * (x - dim_mid) \
            - (y - dim_mid) * (y - dim_mid) \
            - (z - dim_mid) * (z - dim_mid) \
        ) / (2.0 * var * var))

def gen_pdf(var, dim, dim_mid):
    total = 0
    res = []

    for k in xrange(dim):
        for j in xrange(dim):
            for i in xrange(dim):
                norm = norm2d(var, i, j, k, dim_mid)
                res.append(norm)
                total += norm

    for x in xrange(len(res)):
        res[x] /= total

    return res

dim      = 7
dim_mid  = dim / 2;
shape    = [dim, dim, dim]
in_shape = (3,)

PC_GLOBAL_INHIB = 0.00002
PC_VT_INJECT_ENERGY = 0.15
PC_VT_RESTORE = 0.05
VT_ACTIVE_DECAY = 1.0

const_global_inhibit = tf.fill(shape, PC_GLOBAL_INHIB, name="ConstGlobalInhibit")
const_active_decay = tf.fill(shape, VT_ACTIVE_DECAY, name="ConstGlobalDecay")

pose_input = tf.placeholder(tf.int64, shape=in_shape, name="PoseInput")
view_input = tf.placeholder(tf.int64, shape=in_shape, name="ViewInput")
posecells  = tf.Variable(tf.ones(shape, tf.float32), name="PoseCells")
view_decay = tf.Variable(tf.fill(shape, VT_ACTIVE_DECAY), name="ViewDecay")

# Generate PDFs for excite/inhibit spreading
excite_pdf = gen_pdf(1, dim, dim_mid)
inhibit_pdf = gen_pdf(2, dim, dim_mid)

# FIXME: Moving the PDF to known location and using it to shape the normalisation rather than "spread" the energy.
# To roll the PDF like this to spread for every position is a bit much.
# Start with a smaller shape, perhaps can convolve it with TF somehow.

# Roll the PDFs until aligned with current index and reshape
rolled_excite_pdf = tf.py_func(np.roll, [excite_pdf, ((pose_input[2]-dim/2) * dim * dim) + ((pose_input[1]-dim/2) * dim) + ((pose_input[0]-dim/2) + 1)], tf.float32, name="roll_excite")
rolled_inhibit_pdf = tf.py_func(np.roll, [inhibit_pdf, ((pose_input[2]-dim/2) * dim * dim) + ((pose_input[1]-dim/2) * dim) + ((pose_input[0]-dim/2) + 1)], tf.float32, name="roll_inhibit")
shaped_excite_pdf = tf.reshape(rolled_excite_pdf, shape, name="ExcitePDF")
shaped_inhibit_pdf = tf.reshape(rolled_inhibit_pdf, shape, name="InhibitPDF")

# Initalise globals
tf.global_variables_initializer().run(session=sess)

# Add to decay, except for current view
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1043
decay_current = view_decay[view_input[0], view_input[1], view_input[2]]
decay_added = tf.assign(view_decay, tf.add(view_decay, VT_ACTIVE_DECAY))
# Replace current view with original decay
decay_update = tf.scatter_nd_update(view_decay, [view_input], [decay_current])

# Calculate energy given decay
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1047
energy = tf.divide(PC_VT_INJECT_ENERGY * 1.0, tf.multiply(30.0, tf.subtract(30.0, tf.exp(tf.multiply(1.2, decay_update)))), name="CalcEnergy")

# Inject energy given by decay, across all indexes
inject = tf.add(posecells, energy, name="Inject")

# Slightly restore decay
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1063
decay_restore = tf.subtract(view_decay, PC_VT_RESTORE)

# Limit decay to VT_ACTIVE_DECAY
# https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L1065
decay_applied = tf.subtract(decay_restore, const_active_decay, name="DecayRestore")
decay_limited = tf.where(tf.greater_equal(decay_restore, const_active_decay), x=decay_applied, y=const_active_decay, name="DecayLimit")
decay_update = tf.assign(view_decay, decay_limited)

#inject = tf.scatter_nd_add(posecells, [pose_input], [energy], name="Inject")
excite = tf.multiply(inject, shaped_excite_pdf, name="Excite")
inhibit = tf.subtract(excite, tf.multiply(excite, shaped_inhibit_pdf), name="Inhibit")

global_applied = tf.subtract(inhibit, const_global_inhibit, name="GlobalInhibit")
global_limited = tf.where(tf.greater_equal(inhibit, const_global_inhibit), x=global_applied, y=inhibit)

norm_posecells = tf.realdiv(global_limited, tf.reduce_sum(global_limited), name="Norm")
#norm_posecells = tf.realdiv(inhibit, tf.reduce_sum(inhibit), name="Norm")

# Reassign
process_posecells = norm_posecells
process_decays = decay_update
#process = tf.assign(posecells, norm_posecells, name="Process")

def main(_):
    writer = tf.summary.FileWriter("/tmp/tf_posecell_graph", sess.graph)

    res = None

    window = Window()
    window.dim = dim

    def update(dt):
        print(window.x, window.y, window.z)

        res = sess.run([process_posecells, process_decays], feed_dict={
            pose_input: [window.x, window.y, window.z],
            view_input: [window.x, window.y, window.z]
        })
        data = res[0]

        # Re-scale between [0, 1] for rendering (transparency percentage)
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
