from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd as nd
from op import down_sample, class_predictor, box_predictor, flatten_prediction, concat_predictions, stem, dense_block, transition_layer
from mxnet.contrib.ndarray import MultiBoxPrior
def dense_body(block_configs = [6, 8, 8, 8], compression = 0.5, growth_rate = 48):
	out = nn.HybridSequential()
	num_init_features = [64,64,128]
	out.add(stem(num_init_features))
	num_features = 24
	for i, block_config in enumerate(block_configs):
		out.add(dense_block(block_config, 4, 12, 0))
		num_features = num_features + int(block_config*growth_rate)
		if i != len(block_configs) -1:
			out.add(transition_layer(int(num_features*compression)))
			num_features = int(num_features*compression)
	return out
def toy_ssd_model(num_anchors, num_classes):
	body_net = dense_body()
	downsamples = nn.Sequential()
	class_preds = nn.Sequential()
	box_preds = nn.Sequential()

	downsamples.add(down_sample(128))
	downsamples.add(down_sample(128))
	downsamples.add(down_sample(128))

	for scale in range(5):
		class_preds.add(class_predictor(num_anchors, num_classes))
		box_preds.add(box_predictor(num_anchors))
	return body_net, downsamples, class_preds, box_preds

def toy_ssd_forward(x, body, downsamples, class_preds, box_preds, sizes, ratios):
	x = body(x)
	default_anchors = []
	predicted_boxes = []
	predicted_classes = []
	for i in range(5):
		default_anchors.append(MultiBoxPrior(x, sizes[i], ratios = ratios[i]))
		predicted_boxes.append(flatten_prediction(box_preds[i](x)))
		predicted_classes.append(flatten_prediction(class_preds[i](x)))
		#print(predicted_classes[i].shape)
		if i < 3:
			x = downsamples[i](x)
		elif i ==3:
			x = nd.Pooling(x, global_pool = True, pool_type = 'max', kernel = (4,4))
	return default_anchors, predicted_boxes, predicted_classes

class ToySSD(gluon.Block):
	def __init__(self, num_classes, **kwargs):
		super(ToySSD, self).__init__(**kwargs)
		self.anchor_sizes = [[.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
		self.anchor_ratios = [[1, 2, .5]] * 5
		self.num_classes = num_classes
		with self.name_scope():
			self.body, self.downsamples,self.class_preds, self.box_preds = toy_ssd_model(4, num_classes)
	def forward(self, x):
		default_anchors, predicted_boxes, predicted_classes = toy_ssd_forward(x, self.body, self.downsamples, \
			self.class_preds, self.box_preds, self.anchor_sizes, self.anchor_ratios)
		
		anchors = concat_predictions(default_anchors)
		box_preds = concat_predictions(predicted_boxes)
		class_preds = concat_predictions(predicted_classes)
		class_preds = nd.reshape(class_preds, shape = (0,-1, self.num_classes+1))
		#print(class_preds.shape)
		return anchors, class_preds, box_preds
