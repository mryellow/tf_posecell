import math
import tensorflow as tf

from continuous_attractor import ContinuousAttractor
from path_integration import PathIntegration

class Posecells(ContinuousAttractor, PathIntegration):
	"""
    Posecells
	"""

	def __init__(self, sess, dim_th, dim_xy):
		self.sess = sess

		self.dim_th = dim_th
		self.dim_xy = dim_xy
		self.mid_th = self.dim_th / 2
		self.mid_xy = self.dim_xy / 2

		self.shape   = [self.dim_th, self.dim_xy, self.dim_xy]
		self.centre  = [self.mid_th, self.mid_xy, self.mid_xy]
		self.size_th = (2.0 * math.pi) / self.dim_th

		# FIXME: Probably a good idea to pass these in...
		super(Posecells, self).__init__()

		self.cells = tf.Variable(tf.zeros(self.shape, tf.float32), name="PoseCells")

		tf.variables_initializer([self.cells]).run(session=self.sess)

		# Inject some energy at the start.
		tf.assign(self.cells, tf.scatter_nd_add(self.cells, [self.centre], [1]), name="InitPoseCell"
		).eval(session=self.sess)

	def process(self):
		pass
