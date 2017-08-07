class ViewTemplates(object):
	"""
    ViewTemplates
	"""

	def __init__(self):
		self.input = tf.placeholder(tf.int64, shape=in_shape, name="ViewInput")

        # TODO: Actually keeping VTs now...
        self.decay = tf.Variable(tf.fill(shape, VT_ACTIVE_DECAY), name="ViewDecay")

	def process(self):

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

		return decay_restore
