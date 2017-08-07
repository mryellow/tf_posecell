import pyglet
from pyglet.gl import *
from pyglet.window import key

from voxel_engine import VoxelEngine

class Window(pyglet.window.Window):
	"""
    Window
	"""
	is_event_handler = True

	def __init__(self, dim_th, dim_xy):
		super(Window, self).__init__(resizable=True, caption='PyVoxel')

		self.dim_th = dim_th
		self.dim_xy = dim_xy
		self.mid_th = self.dim_th / 2
		self.mid_xy = self.dim_xy / 2

		self.voxel  = VoxelEngine(self.dim_xy, self.dim_xy, self.dim_th)
		glClearColor(0.7, 0.7, 0.8, 1)

		self.trans_accel = 0.2
		self.rot_accel   = 1.0

		self.bindings = {
			key.W: 'up',
			#key.S: 'down',
			#key.A: 'left',
			#key.D: 'right',
			#key.Q: 'turn-left',
			#key.E: 'turn-right',
			key.LEFT: 'left',
			key.RIGHT: 'right',
			key.UP: 'up',
			key.DOWN: 'down',
			#key.I: 'vt1',
			#key.J: 'vt2',
			#key.L: 'vt3',
		}

		# In ROS speak:
		# odo->twist.twist.linear.x, odo->twist.twist.angular.z
		# x in metres, scaled by `vtrans /= PC_CELL_X_SIZE;`
		# z in rad? Actually using degrees then converting.
		self.vtrans = {
			'x': 0.0,
			'y': 0.0,
			'z': 0.0
		}
		self.vrot = {
			'x': 0.0,
			'y': 0.0,
			'z': 0.0
		}

	def on_draw(self):
		self.clear()
		self.setup_3D()
		self.voxel.draw()

	def setup_3D(self):
		""" Setup the 3D matrix """
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(70, self.width / float(self.height), 0.1, 200)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		gluLookAt(24, 20, 20, 0, 10, 4, 0, 1, 0)

	#def on_key_press(self, k, m):
	#	binds = self.bindings
	#	if k in binds:
	#		if binds[k] == 'vt1':
	#			self.view = [1,1,1]
	#		if binds[k] == 'vt2':
	#			self.view = [self.dim-1,self.dim-1,self.dim-1]
	#		if binds[k] == 'vt3':
	#			self.view = [self.dim/2,1,self.dim/2]

	def on_key_release(self, k, m):
		binds = self.bindings
		if k in binds:
			if binds[k] == 'left':
				self.vrot['z'] += self.rot_accel
			if binds[k] == 'right':
				self.vrot['z'] += self.rot_accel
			if binds[k] == 'up':
				self.vtrans['x'] += self.trans_accel
			if binds[k] == 'down':
				# FIXME: Reverse direction o vrot
				if self.vtrans['x'] > 0:
					self.vtrans['x'] -= self.trans_accel

			#if self.vrot['z'] > 359:
			#	self.vrot['z'] = 0
			#if self.vrot['z'] < -359:
			#	self.vrot['z'] = 0

			if self.vtrans['x'] < 0:
				self.vtrans['x'] = 0

			return True
		return False
