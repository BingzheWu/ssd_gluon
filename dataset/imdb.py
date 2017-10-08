import numpy as np

class Imdb(object):

	def __init__(self, name):
		self.name = name
		self.classes = []
		self.num_classes = 0
		self.image_set_index = []
		self.num_images = 0
		self.labels = None
		self.padding = 0
	def image_path_from_index(self, index):
		raise NotImplementedError
	def label_from_index(self, index):
		raise NotImplementedError