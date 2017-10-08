import mxnet as mx
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior
import matplotlib.pyplot as plt
from mxnet.gluon import nn
import mxnet.gluon as gluon
def test_anchor(h = 40, w = 40):
	x = nd.random_uniform(shape = (1,3,h,w))
	y = MultiBoxPrior(x,sizes = [0.5, 0.25, 0.1], ratios = [1, 2, 0.5])

	boxes = y.reshape((h,w,-1,4))
	print('The first anchor box at row 21, column 21:', boxes[20,20,0,:])
	return boxes
def box2rect(box, color, linewidth = 3):
	box = box.asnumpy()
	return plt.Rectangle((box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]), 
		fill = False, edgecolor = color, linewidth = linewidth)
def vis_anchor(colors=['blue', 'green', 'red', 'black','magenta']):
	h = 40
	w=40
	plt.imshow(nd.ones((h,w,3)).asnumpy())
	anchors = test_anchor()[20,20,:,:]
	for i in range(anchors.shape[0]):
		plt.gca().add_patch(box2rect(anchors[i,:]*h, colors[i]))
	plt.show()


def class_predictor(num_anchors, num_classes):
	return nn.Conv2D(num_anchors*(num_classes+1), 3, padding = 1)
def box_predictor(num_anchors):
	return nn.Conv2D(num_anchors*4, 3, padding = 1)
def down_sample(num_filters):
	out = nn.HybridSequential()
	for _ in range(2):
		out.add(nn.Conv2D(num_filters, 3, strides = 1, padding = 1))
		out.add(nn.BatchNorm(in_channels = num_filters))
		out.add(nn.Activation('relu'))
	out.add(nn.MaxPool2D(2))
	return out
def conv_block(num_filters, conv_shape, strides, padding):
	out = nn.HybridSequential()
	out.add(nn.Conv2D(num_filters, conv_shape, strides, padding,))
	out.add(nn.BatchNorm(in_channels = num_filters))
	out.add(nn.Activation('relu'))
	return out
def stem(num_filters = [64, 64, 128]):
	out = nn.HybridSequential()
	for i, num_filter in enumerate(num_filters):
		if i ==0:
			out.add(conv_block(num_filter, 3, 2, 1))
		else:
			out.add(conv_block(num_filter, 3, 1, 1))
	out.add(nn.MaxPool2D(2))
	return out
class dense_layer(gluon.HybridBlock):
	def __init__(self, growth_rate, bn_size, drop_rate):
		super(dense_layer, self).__init__()
		self.out = nn.HybridSequential()
		self.out.add(conv_block(bn_size*growth_rate,1 ,1,0))
		self.out.add(conv_block(growth_rate, 3, 1, 1))
		self.drop_rate = drop_rate
		if self.drop_rate >0:
			self.out.add(nn.Dropout(self.drop_rate))
	def hybrid_forward(self, F, x):
		new_features = self.out(x)
		out = [x, new_features]
		return F.concat(x,new_features,dim = 1)
class dense_block(gluon.HybridBlock):
	def __init__(self, num_layers, bn_size, growth_rate, drop_rate):
		super(dense_block, self).__init__()
		self.out = nn.HybridSequential()
		for i in range(num_layers):
			self.out.add(dense_layer(growth_rate, bn_size, drop_rate))
	def hybrid_forward(self, F, x):
		return self.out(x)
def transition_layer(num_output):
	out = nn.HybridSequential()
	out.add(conv_block(num_output, 1, 1, 0))
	out.add(nn.AvgPool2D(2))
	return out

def flatten_prediction(pred):
	return nd.flatten(nd.transpose(pred, axes = (0, 2, 3, 1)))
def concat_predictions(preds):
	return nd.concat(*preds, dim = 1)



if __name__ == '__main__':
	vis_anchor()