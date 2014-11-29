# IMPORTS
from numpy           import pi, sin, cos, linspace, array, vstack
from numpy           import concatenate as cat
from matplotlib      import pyplot      as plt
from matplotlib      import patches
from matplotlib.path import Path
from psychopy        import visual

# TODOs:
# [ ] draw pizzas not as speparate parts but united according to grouping
# [ ] units used by CircStim?


class CircStim:
	'''
	c = CircStim(**kwargs)

	CircStim is a pizza-like circular stimulus.
	You can check out how it looks like by typing:
	>> c = CircStim(n = 4)
	>> c.plot()

	parameters
	----------

	n       - number of triangular polygons each 
			  pizza element concisits of
	pattern - 2-character string defining contrast
			  pattern of the pizzas
			  first character: 
			  'x' - x-shaped constrast pattern
			  '+' - +-shaped constrast pattern
			  'm' - mask type contrast pattern,
			        each pizza is different
			  second character:
			  'd' or 'b' - the first pizza is dark (black)
			  'l' or 'w' - the first pizza is bright as light (white)
    window  - psychopy window to draw the stimulus in
    opacity - we use opactiy to manipulate contrast
	'''
	
	# defaults
	r       = 1
	n       = 4
	pattern = '+l'
	opacity = 1.0

	rgb     = {'white' : (1, 1, 1),
			   'black' : (0, 0, 0)}

	def __init__(self, **kwargs):
		# set passed attribures
		for name, value in kwargs.items():
			setattr(self, name, value)

		self.get_circle_points()
		self.get_pizza_parts()
		self.apply_pattern()
		self.apply_contrast()


	def get_circle_points(self):
		radians = linspace(0, 2*pi, self.n*8 + 1)
		self.r1points = array([sin(radians), cos(radians)]).transpose()
		self.points = self.r1points * self.r # set radius
		
	def get_pizza_parts(self):
		# last pizza part seems to be obsolete...
		self.pizza = [cat([self.points[i : i + self.n + 1], array([[0.0, 0.0]]) ]) 
						for i in range(0, len(self.points), self.n)]

	def apply_contrast(self):
		w = 0.5 + self.opacity * 0.5
		b = 0.5 - self.opacity * 0.5

		self.rgb['white'] = (w, w, w)
		self.rgb['black'] = (b, b, b)

	def apply_pattern(self):
		# group pizzas
		self.pizza_group = []
		if 'm' in self.pattern:
			self.pizza_group.append(list(range(0,8,2)))
			self.pizza_group.append(list(range(1,8,2)))
		elif '+' in self.pattern:
			self.pizza_group.append([0, 1, 4, 5])
			self.pizza_group.append([2, 3, 6, 7])
		elif 'x' in self.pattern:
			self.pizza_group.append([0, 3, 4, 7])
			self.pizza_group.append([1, 2, 5, 6])

		# set fill color
		if 'd' in self.pattern or 'b' in self.pattern:
			self.pizza_fill = ['black', 'white']
		elif 'l' in self.pattern or 'w' in self.pattern:
			self.pizza_fill = ['white', 'black']


	def create_shapes(self):
		self.shapes = [0] * 8
		for i, grp in enumerate(self.pizza_group):
			for p in grp:
				self.shapes[p] = visual.ShapeStim(self.window, 
					                              lineWidth  = 0, 
					                              fillColor  = self.pizza_fill[i], 
					                              lineColor  = self.pizza_fill[i], 
					                              vertices   = self.pizza[p], 
					                              closeShape = True, 
					                              pos        = (0, 0), 
					                              opacity    = self.opacity
					                              )

	def draw(self):
		for shp in self.shapes:
			shp.draw()

	def plot(self):

		figure = plt.figure()
		ax     = figure.add_subplot(1, 1, 1, axisbg = 'grey')

		for n in range(len(self.pizza)-1):
			self.plot_pizza(ax, n)

		plt.axis('equal')
		plt.xlim((-2, 2))
		plt.ylim((-2, 2))
		plt.xticks((), ())
		plt.yticks((), ())

		plt.show()

	def plot_pizza(self, ax, n):

		# vertices:
		vert = self.pizza[n]
		vert = vstack((vert, vert[0]))

		# path:
		pth = [Path.MOVETO] + [Path.LINETO] * (self.n + 1) + [Path.CLOSEPOLY]

		# plot
		path  = Path(vert, pth)
		fccol = self.pizza_fill[0] if n in self.pizza_group[0] else self.pizza_fill[1]
		col = self.rgb[fccol]
		patch = patches.PathPatch(path, facecolor = col, lw = 0.5, edgecolor = col)
		ax.add_patch(patch)

