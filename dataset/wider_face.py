import os
import numpy as np
from imdb import Imdb
import cv2

class WiderFace(Imdb):

	def __init__(self, data_path = '/home/zhou/wider_face/images', \
	 	gt_txt_file = '/home/zhou/wider_face/wider_face_train_bbx_gt.txt',\
		shuffle = False, is_train = True):

		super(WiderFace, self).__init__('wider_face')
		self.classes = ['face']
		self.data_path = data_path
		self.is_train = is_train
		self.gt_txt_file = gt_txt_file
		self.num_classes = len(self.classes)
		self.image_set_index = self._load_image_set_index(shuffle)
		self.num_images  = len(self.image_set_index)
		self.padding = 60
		if self.is_train:
			self.labels = self._load_image_labels()

	def _load_image_set_index(self, shuffle):
		with open(self.gt_txt_file) as f:
			image_set_index = [x.strip() for x in f.readlines() if x.strip().endswith('.jpg')]
		if shuffle:
			np.random.shuffle(image_set_index)
		return image_set_index
	def _load_image_labels(self):
		temp = []
		gt_dict = {}
		max_objects = 0
		start_idx = 0
		end_idx = 0
		labels = []
		temp = []
		num_images = 0
		image_set_index = []
		with open(self.gt_txt_file) as f:
			data = f.readlines()
			while start_idx < len(data):
				image_index = data[start_idx].strip()
				image_path = os.path.join(self.data_path, image_index)
				box_num = int(data[start_idx+1].strip())
				end_idx = start_idx+1+box_num
				label = []
				#if box_num > self.padding:
			#		continue
				img = cv2.imread(image_path)
				height, width, c = img.shape
				assert image_index.endswith('jpg')
				#print(image_path)
				for box_idx in range(start_idx+2, end_idx+1):
					cls_id = 0
					xmin, ymin, w, h, blur, expression, ill, invalid, occlusion, pose= data[box_idx].strip().split(' ')
					scale_xmin = float(xmin) / width
					scale_ymin = float(ymin) / height
					scale_xmax = float(int(xmin)+int(w)) / width
					scale_ymax = float(int(ymin)+int(h)) / height
					if box_idx-start_idx+1 > self.padding:
						break
					if blur or not invalid:
						label.append([cls_id, scale_xmin, scale_ymin, scale_xmax, scale_ymax])
				start_idx = end_idx+1
				if label!=[]:
					image_set_index.append(image_index)
					temp.append(np.array(label))
					num_images += 1
			labels = []
			for l in temp:
				l = np.lib.pad(l, ((0, self.padding-l.shape[0]),(0,0)), 'constant', constant_values = (-1,-1))
				labels.append(l)	
		self.image_set_index = image_set_index
		self.num_images = num_images
		return np.array(labels)
	def image_path_from_index(self, index):
		name = self.image_set_index[index]
		image_file = os.path.join(self.data_path, name)
		return image_file
	def label_from_index(self, index):
		return self.labels[index]

if __name__ == '__main__':
	face_imdb = WiderFace()
	face_imdb._load_image_labels()