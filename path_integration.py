import math
import tensorflow as tf

PC_CELL_X_SIZE = 1.0

class PathIntegration(object):
	"""
    PathIntegration
	"""

	def __init__(self):
		super(PathIntegration, self).__init__()

		self.vtrans = tf.placeholder(tf.float32, shape=[], name="vtransInput")
		self.vrot   = tf.placeholder(tf.float32, shape=[], name="vrotInput")

		self.size_th  = (2.0 * math.pi) / self.dim_th;

		#tf.variables_initializer([self.vtrans, self.vrot]).run(session=self.sess)

		rots_idx, rots_val   = self.rotation()
		trans_idx, trans_val = self.translation()

		self.trans_weights_dense = tf.sparse_to_dense(
		    sparse_indices=trans_idx,
		    output_shape=[self.dim_th, self.dim_xy, self.dim_xy, self.dim_th, self.dim_xy, self.dim_xy],
		    sparse_values=trans_val,
		    default_value=0,
		    validate_indices=False
		)
		self.rots_weights_dense = tf.sparse_to_dense(
		    sparse_indices=rots_idx,
		    output_shape=[self.dim_th, self.dim_xy, self.dim_xy, self.dim_th, self.dim_xy, self.dim_xy],
		    sparse_values=rots_val,
		    default_value=0,
		    validate_indices=False
		)

	def translation(self):
		print('Generating Translation weights...')
		val = []
		idx = []
		cnt = 0

		# TODO: Scale velocity by size of time step (not really applicable to DQN agents which step evenly)
		# `vtrans = vtrans * time_diff_s;`
		# `vrot = vrot * time_diff_s;`

		vtranscell = tf.divide(self.vtrans, PC_CELL_X_SIZE)

		# Z axis
		for k in xrange(self.dim_th):
			trans_dir = k * self.size_th #th index * size of each th in rad + angle_to_add (for reverse)

			# Break down to one single quadrant
			# `math.floor(trans_dir * 2 / math.pi)` gives number of quadrant.

			trans_quad = math.floor(trans_dir * 2 / math.pi)
			trans_dir = trans_dir - trans_quad * math.pi / 2

			# Y axis
			for j in xrange(self.dim_xy):
				# X axis
				for i in xrange(self.dim_xy):
					weights = []

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
					def wrap(quad):
						for card in xrange(len(quad)):
							if quad[card] < 0:
								quad[card] = self.dim_xy-1
							if quad[card] > self.dim_xy-1:
								quad[card] = 0

					wrap(ne)
					wrap(nw)
					wrap(se)
					wrap(sw)

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

	def rotation(self):
		print('Generating Rotation weights...')
		val = []
		idx = []

		# Convert to radians
		rads = tf.multiply(self.vrot, math.pi/180)

		weight = tf.mod(tf.divide(tf.abs(rads), self.size_th), 1)
		weight = tf.where(tf.equal(weight, 0.0), 1.0, weight)

		# TODO: `tf.sign()`
		sign_vrot = tf.cond(rads < 0, lambda: 1.0, lambda: -1.0)

		shifty1 = tf.cast(tf.multiply(sign_vrot, tf.floor(tf.divide(tf.abs(rads), self.size_th))), dtype=tf.int32)
		shifty2 = tf.cast(tf.multiply(sign_vrot, tf.ceil(tf.divide(tf.abs(rads), self.size_th))), dtype=tf.int32)

		cond = lambda i: tf.greater(i, 0)
		body = lambda i: tf.subtract(i, self.dim_th)
		tf.while_loop(cond, body, [shifty1])
		tf.while_loop(cond, body, [shifty2])

		for k in xrange(self.dim_th):
			newk1 = tf.mod(tf.subtract(k, shifty1), self.dim_th)
			newk2 = tf.mod(tf.subtract(k, shifty2), self.dim_th)
			for j in xrange(self.dim_xy):
				for i in xrange(self.dim_xy):
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

	def path_integration(self):
		posecells_reshaped = tf.reshape(self.cells, [-1, 1, 1, self.dim_th, self.dim_xy, self.dim_xy])
		translate = tf.tensordot(posecells_reshaped, self.trans_weights_dense, axes=3, name="Translate")
		rotate    = tf.tensordot(translate, self.rots_weights_dense, axes=3, name="Rotate")
		path_integration = tf.squeeze(rotate)

		return tf.assign(self.cells, path_integration)
