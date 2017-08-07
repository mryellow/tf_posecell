import math
import numpy as np
import tensorflow as tf

PC_GLOBAL_INHIB = 0.00002

class ContinuousAttractor(object):
	"""
    ContinuousAttractor
	"""

	def __init__(self):
		super(ContinuousAttractor, self).__init__()

		# Generate self-connection weights
		self.excite_pdf = self.pdf(1)
		self.inhibi_pdf = self.pdf(2)

		self.excite_weights = tf.constant(self.weights(self.excite_pdf), dtype=tf.float32, name="WeightsExcite")
		self.inhibi_weights = tf.constant(self.weights(self.inhibi_pdf), dtype=tf.float32, name="WeightsInhibit")

		#tf.variables_initializer([self.excite_weights, self.inhibi_weights]).run(session=self.sess)

	def norm2d(self, var, x, y, z):
	    return 1.0 / (var * math.sqrt(2.0 * math.pi)) \
	        * math.exp(( \
	            - (x - self.mid_xy) * (x - self.mid_xy) \
	            - (y - self.mid_xy) * (y - self.mid_xy) \
	            - (z - self.mid_th) * (z - self.mid_th) \
	        ) / (2.0 * var * var))

	def pdf(self, var):
	    total = 0
	    res = []

	    for k in xrange(self.dim_th):
	        for j in xrange(self.dim_xy):
	            for i in xrange(self.dim_xy):
	                norm = self.norm2d(var, i, j, k)
	                res.append(norm)
	                total += norm

	    for x in xrange(len(res)):
	        res[x] /= total

	    return res

	def weights(self, pdf):
	    print('Generating PDF weights...')
	    res = []
	    for k in xrange(self.dim_th):
	        res.append([])
	        for j in xrange(self.dim_xy):
	            res[k].append([])
	            for i in xrange(self.dim_xy):
	                # TODO: Sparsify, indexes and values, over a min limit
	                res[k][j].append(self.displace(pdf, k, j, i))

	    return np.asarray(res)

	def displace(self, pdf, x, y, z):
	    # Fully connected
	    res = np.roll(pdf, (
			(x-self.mid_xy) * self.dim_xy * self.dim_th) + \
			((y-self.mid_xy) * self.dim_th) + \
			(z-self.mid_th)
		)
	    # TODO: Locally connected "Activity Packets"
	    # [[left, most, corner], [smaller, p, d, f]]
	    res = np.reshape(res, self.shape)
	    return res

	def excite(self):
		# Local Excite
		posecells_reshaped = tf.reshape(self.cells, [-1, 1, 1, self.dim_th, self.dim_xy, self.dim_xy])
		excite = tf.squeeze(tf.tensordot(posecells_reshaped, self.excite_weights, axes=3, name="Excite"))
		#res = sess.run(excite)
		#print('excite', res, res.shape)

		# Local Inhibit
		# FIXME: Reshape needed? Keep the shape and reshape once? Well just don't `squeeze` `excite` first.
		excite_reshaped = tf.reshape(excite, [-1, 1, 1, self.dim_th, self.dim_xy, self.dim_xy])
		inhibi = tf.subtract(excite, tf.squeeze(tf.tensordot(excite_reshaped, self.inhibi_weights, axes=3, name="Inhibit")))
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

		return tf.assign(self.cells, norm_posecells)
