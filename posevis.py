#import random

import pyglet
from pyglet.gl import *
from pyglet.window import key

class VoxelEngine:
	def __init__(self, w, h, d):
		""" Create a new voxel engine. """
		self.w = w
		self.h = h
		self.d = d
		# Create the 3D array
		self.voxels = []
		for _ in range(self.w):
			self.voxels.append([[0 for _ in range(self.d)] for _ in range(self.h)])

	def set(self, x, y, z, value):
		""" Set the value of the voxel at position (x, y, z). """
		self.voxels[x][y][z] = value

	def draw(self):
		""" Draw the voxels. """
		vertices = (
			0, 0, 0,	# vertex 0
			0, 0, 1,	# vertex 1
			0, 1, 0,	# vertex 2
			0, 1, 1,	# vertex 3
			1, 0, 0,	# vertex 4
			1, 0, 1,	# vertex 5
			1, 1, 0,	# vertex 6
			1, 1, 1,	# vertex 7
		)
		indices = (
			0, 1, 3, 2,	# top face
			4, 5, 7, 6,	# bottom face
			0, 4, 6, 2,	# left face
			1, 5, 7, 3,	# right face
			0, 1, 5, 4,	# down face
			2, 3, 7, 6, # up face
		)
		colors = (
			(107, 83, 28), # dirt
			(18, 124, 39), # grass
			(168, 142, 95), # wood
			(88, 181, 74), # leaves
		)

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);

		# Loop through each voxel
		voxel_min = 1
		voxel_max = 0
		for x in range(self.w):
			for y in range(self.h):
				for z in range(self.d):
					voxel_min = min(voxel_min, self.voxels[x][y][z])
					voxel_max = max(voxel_max, self.voxels[x][y][z])

		for x in range(self.w):
			for y in range(self.h):
				for z in range(self.d):
					voxel_type = self.voxels[x][y][z]
					if voxel_type != 0:
						glTranslated(x, y, z)
						normalized = (voxel_type-voxel_min)/(voxel_max-voxel_min)
						alpha = 0
						if normalized > 0:
							alpha = int(155*normalized)
						glColor4ub(18, 124, 39, alpha)
						pyglet.graphics.draw_indexed(8, GL_QUADS,
							indices, ('v3i', vertices))
						glTranslated(-x, -y, -z)


class Window(pyglet.window.Window):
	"""
    WorldLayer
	"""
	is_event_handler = True

	def __init__(self):
		super(Window, self).__init__(resizable=True, caption='PyVoxel')
		self.voxel = VoxelEngine(20, 25, 20)
		glClearColor(0.7, 0.7, 0.8, 1)

		self.bindings = {
			key.W: 'up',
			key.S: 'down',
			key.A: 'left',
			key.D: 'right',
			key.Q: 'turn-left',
			key.E: 'turn-right',
			key.LEFT: 'left',
			key.RIGHT: 'right',
			key.UP: 'up',
			key.DOWN: 'down',
			key.I: 'vt1',
			key.J: 'vt2',
			key.L: 'vt3',
		}
		#buttons = {}
		#for k in self.bindings:
		#	buttons[self.bindings[k]] = 0
		#self.buttons = buttons

		self.dim = None
		self.pose = [0,0,0]
		self.view = [0,0,0]

		# In ROS speak:
		# odo->twist.twist.linear.x, odo->twist.twist.angular.z
		# x in metres, scaled by `vtrans /= PC_CELL_X_SIZE;`
		# z in rad
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

		self.pose_last = self.pose[:]
		self.view_last = self.view[:]

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

	def on_key_release(self, k, m):
		binds = self.bindings
		if k in binds:
			if binds[k] == 'left':
				self.pose[0] += 1
			#	self.vrot['z'] -= 1
			if binds[k] == 'right':
				self.pose[0] -= 1
			#	self.vrot['z'] += 1
			if binds[k] == 'up':
				self.pose[1] += 1
				self.vtrans['x'] += 1
			if binds[k] == 'down':
				self.pose[1] -= 1
				# FIXME: Reverse direction o vrot
				if self.vtrans['x'] > 0:
					self.vtrans['x'] -= 1
			if binds[k] == 'turn-left':
				self.pose[2] += 1
				self.vrot['z'] -= 1
			if binds[k] == 'turn-right':
				self.pose[2] -= 1
				self.vrot['z'] += 1

			if self.vrot['z'] > 359:
				self.vrot['z'] = 0
			if self.vrot['z'] < -359:
				self.vrot['z'] = 0

			if binds[k] == 'vt1':
				self.view = [1,1,1]
			if binds[k] == 'vt2':
				self.view = [3,4,5]
			if binds[k] == 'vt3':
				self.view = [6,6,6]

			self.pose[0] %= self.dim
			self.pose[1] %= self.dim
			self.pose[2] %= self.dim
			#print('on_key_release', self.pose[0], self.pose[1], self.pose[2])
			return True
		return False

if __name__ == '__main__':
    window = Window()
    pyglet.app.run()
