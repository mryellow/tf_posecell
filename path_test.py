import math
import numpy as np
import tensorflow as tf

import pyglet
import random
from pyglet.gl import *
from posevis import Window

sess = tf.Session()

dim      = 7
dim_mid  = dim / 2;
shape    = [dim, dim, dim]
in_shape = (3,)
th_size  = (2.0 * math.pi) / dim;

vtrans = tf.placeholder(tf.float32, shape=[], name="vtransInput")
vrot = tf.placeholder(tf.float32, shape=[], name="vrotInput")

def gen_trans_weights():
    val = []
    idx = []
    cnt = 0

    for k in xrange(dim):
        #res.append([])
        for j in xrange(dim):
            #res[k].append([])
            for i in xrange(dim):

                # TODO:
                # `i` or is it `k`, is z index for trans_dir
                trans_dir = i * th_size #th index * size of each th in rad

                weights = []

                # weight_sw = vtrans * vtrans * cos(dir90) * sin(dir90);
                weights.append(vtrans * vtrans * tf.cos(trans_dir) * tf.sin(trans_dir))
                weight_val = weights[0]
                weight_idx = [k,j,i,k,j,i]
                idx.append(weight_idx)
                val.append(weight_val)

                # Wrap edges
                ii = i+1
                if ii >= dim-1:
                    ii = 0

                # weight_se = vtrans * sin(dir90) * (1.0 - vtrans * cos(dir90));
                weights.append(vtrans * tf.sin(trans_dir) * (1.0 - vtrans * tf.cos(trans_dir)))
                weight_val = weights[1]
                weight_idx = [k,j,i,k,j,ii]
                idx.append(weight_idx)
                val.append(weight_val)

                # Wrap edges
                jj = i+1
                if jj >= dim-1:
                    jj = 0

                # weight_nw = vtrans * cos(dir90) * (1.0 - vtrans * sin(dir90));
                weights.append(vtrans * tf.cos(trans_dir) * (1.0 - vtrans * tf.sin(trans_dir)))
                weight_val = weights[2]
                weight_idx = [k,j,i,k,jj,i]
                idx.append(weight_idx)
                val.append(weight_val)

                # Wrap edges
                ii = i+1
                if ii >= dim-1:
                    ii = 0
                jj = i+1
                if jj >= dim-1:
                    jj = 0

                # weight_ne = 1.0 - weight_sw - weight_se - weight_nw;
                weight_val = 1 - weights[2] - weights[1] - weights[0]
                weight_idx = [k,j,i,k,jj,ii]
                idx.append(weight_idx)
                val.append(weight_val)

    return idx, val

posecells  = tf.Variable(tf.ones(shape, tf.float32), name="PoseCells")

# Initalise globals
tf.global_variables_initializer().run(session=sess)

trans_idx, trans_val = gen_trans_weights()

trans_weights_dense = tf.sparse_to_dense(
    sparse_indices=trans_idx,
    output_shape=[dim, dim, dim, dim, dim, dim],
    sparse_values=trans_val,
    default_value=0,
    validate_indices=False
)
#res = sess.run(trans_weights_dense, feed_dict={
#    vtrans: 10.0,
#    vrot: 10.0
#})
#print('trans_weights_dense', res, res.shape)

# pca_new_rot_ptr[0][0] * weight_ne + pca_new_rot_ptr[0][PC_DIM_XY + 1] * weight_se + pca_new_rot_ptr[PC_DIM_XY + 1][0] * weight_nw;

posecells_reshaped = tf.reshape(posecells, [-1, 1, 1, dim, dim, dim])
trans = tf.squeeze(tf.tensordot(posecells_reshaped, trans_weights_dense, axes=3, name="Excite"))
res = sess.run(trans, feed_dict={
    vtrans: 10.0,
    vrot: 10.0
})
print('trans', res, res.shape)

def main(_):
    pass


if __name__ == '__main__':
    tf.app.run(main=main)
