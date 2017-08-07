import pyglet
from pyglet.gl import *

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
