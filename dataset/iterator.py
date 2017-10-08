import mxnet as mx
import numpy as np
import cv2
from mxnet import ndarray as nd 
class DetIter(mx.io.DataIter):

	def __init__(self, imdb, batch_size, data_shape, mean_pixels = [128, 128, 128], rand_samples=[], \
		rand_mirror = False, shuffle = False, rand_seed = None, \
		is_train = True, max_crop_trial = 50):
		super(DetIter, self).__init__()
		self.batch_size = batch_size
		if isinstance(data_shape, int):
			data_shape = (data_shape, data_shape)
		self._data_shape = data_shape
		self._mean_pixels = mx.nd.array(mean_pixels).reshape((3,1,1))
		self.is_train = is_train
		self._rand_mirror = rand_mirror
		self._shuffle = shuffle
		self._max_crop_trial = max_crop_trial
		self._current = 0
		self.imdb = imdb
		self._size = imdb.num_images
		self._index = np.arange(self._size)
		self._data = None
		self._label = None
		self._get_batch()
	@property

	def provide_data(self):
		return [(k, v.shape) for k, v in self._data.items()]

	@property

	def provide_label(self):
		return [(k, v.shape) for k, v in self._label.items()]

	def reset(self):
		self._current = 0
		if self._shuffle:
			np.random.shuffle(self._index)
	def iter_next(self):
		return self._current < self._size
	def next(self):
		if self.iter_next():
			self._get_batch()
			data_batch = mx.io.DataBatch(data = self._data.values(),
				label = self._label.values(),
				pad = self.getpad(), index = self.getindex())
			self._current += self.batch_size
			return data_batch
		else:
			raise StopIteration
	def getindex(self):
		pad = self._current // self.batch_size
	def getpad(self):
		pad = self._current + self.batch_size -self._size
		return 0 if pad <0 else pad
	def _get_batch(self):
		batch_data = mx.nd.zeros((self.batch_size, 3, self._data_shape[0], self._data_shape[1]))
		batch_label = []
		for i in range(self.batch_size):
			if (self._current + i)>= self._size:
				if not self.is_train:
					continue
				idx = (self._current + i + self._size//2)%self._size
				index = self._index[idx]
			else:
				index = self._index[self._current + i]
			img_path = self.imdb.image_path_from_index(index)
			with open(img_path, 'rb') as fp:
				img_content = fp.read()
			img = mx.img.imdecode(img_content)
			gt = self.imdb.label_from_index(index) if self.is_train else None
			data, label = self.preprocess(img, gt)
			batch_data[i] = data
			if self.is_train:
				batch_label.append(label)
			else:
				self._label = {'label': None}
		self._data = {'data': batch_data}
		if self.is_train:
			self._label = {'label': mx.nd.array(np.array(batch_label))}
		else:
			self._label = {'label': None}
	def preprocess(self, data, label):
		data = mx.img.imresize(data, self._data_shape[1], self._data_shape[0])
		data = mx.nd.transpose(data, (2, 0, 1))
		data = data.astype('float32')
		data = data/255.0
		#data = data - self._mean_pixels
		#label = label[:,:5]
		tmp = []
		for idx, l in enumerate(label):
			tmp.append(l)
		return data, np.array(tmp)


