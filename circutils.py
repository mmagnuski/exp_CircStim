# Class responsible for presentation of circle targets and masks
from numpy      import pi, sin, cos, linspace, array
from numpy      import concatenate as cat
from matplotlib import pyplot      as plt
from psychopy   import visual

# CHECK slider class - if available in psychopy


class CircStim:

	# how CircStim should work:
	#   - contrast should be available to change
	#   - r = radius (what units?)
	#   - pattern = 'x' '+' or 'm'
	#   - win - pass a window object
	#   - 'd'/'l', 0/1 defines whether starting pizza element is dark or white
	#   - n = (minimum 8), divisible by 8; number of angles of the circle
	#     OR n = 1 (positive integer) - multiplied by 8 gives n in the sense above
	#   - df = DataFrame; trial = N; [?]
	r = 1
	n = 4
	pattern = '+l'

	def __init__(self, **kwargs):
		# set passed attribures
		for name, value in kwargs.items():
			setattr(self, name, value)

		self.get_circle_points()
		self.get_pizza_parts()


	def get_circle_points(self):
		radians = linspace(0, 2*pi, self.n*8 + 1)
		self.r1points = array([sin(radians), cos(radians)]).transpose()
		self.points = self.r1points * self.r # set radius
		
	def get_pizza_parts(self):
		self.pizza = [cat([self.points[i : i + self.n + 1], array([[0.0, 0.0]]) ]) 
						for i in range(0, len(self.points), self.n)]

	def apply_pattern(self):
		# group pizzas
		self.pizza_pattern = []
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
					                              vertices   = self.pizza[p], 
					                              closeShape = True, 
					                              pos        = (0, 0), 
					                              opacity    = 0.0
					                              )

	def draw(self):
		for shp in self.shapes:
			shp.draw()

