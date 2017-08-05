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
PC_C_SIZE_TH  = (2.0 * math.pi) / dim;

PC_GLOBAL_INHIB = 0.00002
PC_VT_INJECT_ENERGY = 0.15
PC_VT_RESTORE = 0.05
VT_ACTIVE_DECAY = 1.0
PC_CELL_X_SIZE = 1.0

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
    print('Generating PDF weights...')
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

## Translation
vtrans = tf.placeholder(tf.float32, shape=[], name="vtransInput")

def gen_trans_weights():
    print('Generating Translation weights...')
    val = []
    idx = []
    cnt = 0

    vtranscell = tf.divide(vtrans, PC_CELL_X_SIZE)

    # Z axis
    for k in xrange(dim):
        trans_dir = k * PC_C_SIZE_TH #th index * size of each th in rad + angle_to_add (for reverse)

        # Break down to one single quadrant
        # `math.floor(trans_dir * 2 / math.pi)` gives number of quadrant.

        # FIXME: Not rotating, so must implement each quadrant.
        # FIXME: Currently each is fighting against the next.
        trans_quad = math.floor(trans_dir * 2 / math.pi)
        trans_dir = trans_dir - trans_quad * math.pi / 2

        # Y axis
        for j in xrange(dim):
            # X axis
            for i in xrange(dim):
                weights = []

                # FIXME: Diverges at first... Just taking time to be normalised?
                if trans_quad == 3.0:
                    ne = [j, i]
                    nw = [j-1, i]
                    se = [j, i+1]
                    sw = [j-1, i+1]
                elif trans_quad == 2.0:
                    ne = [j, i]
                    nw = [j+1, i]
                    se = [j+1, i]
                    sw = [j+1, i+1]
                elif trans_quad == 1.0:
                    ne = [j, i]
                    nw = [j+1, i]
                    se = [j, i-1]
                    sw = [j+1, i-1]
                else:
                    ne = [j, i]
                    nw = [j, i-1]
                    se = [j-1, i]
                    sw = [j-1, i-1]

                # Wrap edges
                for card in xrange(len(ne)):
                    if ne[card] < 0:
                        ne[card] = dim-1
                    if ne[card] > dim-1:
                        ne[card] = 0
                for card in xrange(len(nw)):
                    if nw[card] < 0:
                        nw[card] = dim-1
                    if nw[card] > dim-1:
                        nw[card] = 0
                for card in xrange(len(se)):
                    if se[card] < 0:
                        se[card] = dim-1
                    if se[card] > dim-1:
                        se[card] = 0
                for card in xrange(len(sw)):
                    if sw[card] < 0:
                        sw[card] = dim-1
                    if sw[card] > dim-1:
                        sw[card] = 0

                # Note: At 45deg, 0.2 comes from either side and 0.5 from behind.

                # weight_sw = vtranscell * vtranscell * cos(dir90) * sin(dir90);
                weights.append(vtranscell * vtranscell * tf.cos(trans_dir) * tf.sin(trans_dir))
                weight_val = weights[0]
                weight_idx = [k,j,i,k,sw[0],sw[1]]
                idx.append(weight_idx)
                val.append(weight_val)

                # weight_se = vtranscell * sin(dir90) * (1.0 - vtranscell * cos(dir90));
                weights.append(vtranscell * tf.sin(trans_dir) * (1.0 - vtranscell * tf.cos(trans_dir)))
                weight_val = weights[1]
                weight_idx = [k,j,i,k,se[0],se[1]]
                idx.append(weight_idx)
                val.append(weight_val)

                # weight_nw = vtranscell * cos(dir90) * (1.0 - vtranscell * sin(dir90));
                weights.append(vtranscell * tf.cos(trans_dir) * (1.0 - vtranscell * tf.sin(trans_dir)))
                weight_val = weights[2]
                weight_idx = [k,j,i,k,nw[0],nw[1]]
                idx.append(weight_idx)
                val.append(weight_val)

                # weight_ne = 1.0 - weight_sw - weight_se - weight_nw;
                weight_val = 1.0 - weights[2] - weights[1] - weights[0]
                weight_idx = [k,j,i,k,ne[0],ne[1]]
                idx.append(weight_idx)
                val.append(weight_val)

    return idx, val

## Rotation
vrot = tf.placeholder(tf.float32, shape=[], name="vrotInput")

def gen_rot_weights():
    print('Generating Rotation weights...')
    val = []
    idx = []

    # Convert to radians
    rads = tf.multiply(vrot, math.pi/180)

    #weight = mod(abs(vrot)/pc.PC_C_SIZE_TH, 1);
    weight = tf.mod(tf.divide(tf.abs(rads), PC_C_SIZE_TH), 1)
    #weight = tf.divide(tf.abs(rads), PC_C_SIZE_TH)
    #cond = lambda i: tf.greater(i, 1.0)
    #body = lambda i: tf.subtract(i, 1.0)
    #tf.while_loop(cond, body, [weight])
    weight = tf.where(tf.equal(weight, 0.0), 1.0, weight)

    #print(weight.eval(session=sess, feed_dict={
    #    vrot: 1.0
    #}))

    sign_vrot = tf.cond(rads < 0, lambda: 1.0, lambda: -1.0)

    shifty1 = tf.cast(tf.multiply(sign_vrot, tf.floor(tf.divide(tf.abs(rads), PC_C_SIZE_TH))), dtype=tf.int32)
    shifty2 = tf.cast(tf.multiply(sign_vrot, tf.ceil(tf.divide(tf.abs(rads), PC_C_SIZE_TH))), dtype=tf.int32)

    cond = lambda i: tf.greater(i, 0)
    body = lambda i: tf.subtract(i, dim)
    tf.while_loop(cond, body, [shifty1])
    tf.while_loop(cond, body, [shifty2])

    #print(shifty1.eval(session=sess, feed_dict={
    #    vrot: 1.0
    #}))
    #print(shifty2.eval(session=sess, feed_dict={
    #    vrot: 1.0
    #}))

    for k in xrange(dim):
        newk1 = tf.mod(tf.subtract(k, shifty1), dim)
        newk2 = tf.mod(tf.subtract(k, shifty2), dim)
        #print(newk1.eval(session=sess, feed_dict={
        #    vrot: 1.0
        #}))
        for j in xrange(dim):
            for i in xrange(dim):
                #posecells[k][j][i] = pca_new[newk1][j][i] * (1.0 - weight) + pca_new[newk2][j][i] * weight;

                weight_val = 1.0 - weight
                weight_idx = [k,j,i,newk1,j,i]
                idx.append(weight_idx)
                val.append(weight_val)

                weight_val = weight
                weight_idx = [k,j,i,newk2,j,i]
                idx.append(weight_idx)
                val.append(weight_val)

    return idx, val

rots_idx, rots_val = gen_rot_weights()
#res = sess.run(rots_val, feed_dict={
#    vtrans: 10.0,
#    vrot: 45.0
#})
#print('rots_val', res) #, res.shape
#exit()

trans_idx, trans_val = gen_trans_weights()

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

#path_weights_dense = tf.multiply(trans_weights_dense, rots_weights_dense)

#res = sess.run(trans_weights_dense, feed_dict={
#    vtrans: 10.0,
#    vrot: 1.0
#})
#print('trans_weights_dense', res, res.shape)
#exit()

# pca_new_rot_ptr[0][0] * weight_ne + pca_new_rot_ptr[0][PC_DIM_XY + 1] * weight_se + pca_new_rot_ptr[PC_DIM_XY + 1][0] * weight_nw;

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

# Inject some energy at the start. (Why is this needed?)
one_time = tf.assign(posecells, tf.scatter_nd_add(posecells, [[dim_mid,dim_mid,dim_mid]], [1]), name="OneTime")

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
        if window.view_last != window.view:
            print('view', window.view)


        #if window.timestep % 10 == 0:
        #tasks.append(on_view_template)
        #tasks.append(decay_restore)
        tasks.append(process)
        tasks.append(on_odo)

        res = sess.run(tasks, feed_dict={
            #pose_input: window.pose,
            view_input: window.view,
            vtrans: window.vtrans['x'],
            vrot: window.vrot['z']
        })

        #res = sess.run(path_weights_dense, feed_dict={
        #    vtrans: window.vtrans['x'],
        #    vrot: 45 * math.pi/180
        #})
        #print('path_weights_dense', res, res.shape)

        #window.vrot['z']

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
                    window.voxel.set(x, y, z, data[x][y][z])

    pyglet.clock.schedule(update)

    pyglet.app.run()
    writer.close()

if __name__ == '__main__':
    tf.app.run(main=main)
