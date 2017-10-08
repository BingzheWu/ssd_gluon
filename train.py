from loss import SmoothL1Loss, FocalLoss
from models import ToySSD
import mxnet as mx
import time
import mxnet.gluon as gluon
from mxnet import autograd as ag
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxTarget
import sys
sys.path.append('./dataset')
from iterator import DetIter
from wider_face import WiderFace
from toy import get_iterators
def get_iter(name = 'wider_face'):
	imdb = WiderFace()
	train_iter = DetIter(imdb, batch_size = 32, data_shape = 300)
	#train_iter = mx.io.PrefetchingIter(train_iter)
	return imdb, train_iter
def training_targets(default_anchors, class_predicts, labels):
    class_predicts = nd.transpose(class_predicts, axes=(0,2,1))
    z = MultiBoxTarget(default_anchors, labels, class_predicts)
    box_target = z[0]
    box_mask = z[1]
    cls_target = z[2]
    return box_target, box_mask, cls_target

def train():
	cls_metric = mx.metric.Accuracy()
	box_metric = mx.metric.MAE()
	## data load
	imdb, train_data = get_iter()
	#train_data, test_data, class_names, num_class = get_iterators(256, 64)
	#print(num_class)
	## env setup
	ctx = mx.gpu(0)
	try:
		_ = mx.nd.zeros(1, ctx = ctx)
		#train_data.reshape(label_shape = (3,5))
		#train_data = test_data.sync_label_shape(train_data)
	except mx.base.MXNetError as err:
		ctx = mx.cpu()
	## training set
	num_class = 1
	cls_loss = FocalLoss()
	box_loss = SmoothL1Loss()
	net = ToySSD(num_class)
	net.initialize(mx.init.Xavier(magnitude = 2), ctx = ctx)
	net.load_params('models/ssd_%d.params' % 115, ctx)
	net.collect_params().reset_ctx(ctx)
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 1e-3, 'wd': 5e-4})
	epochs = 300
	batch_size = 40
	log_interval = 50
	start_epoch = 0
	for epoch in range(start_epoch, epochs):
		train_data.reset()
		cls_metric.reset()
		box_metric.reset()
		tic = time.time()

		for i, batch in enumerate(train_data):
			btic = time.time()
			with ag.record():
				x = batch.data[0].as_in_context(ctx)
				y = batch.label[0].as_in_context(ctx)
				default_anchors, class_predictions, box_predictions = net(x)
				box_target, box_mask, cls_target = training_targets(default_anchors, class_predictions, y)
				loss1 = cls_loss(class_predictions, cls_target)
				loss2 = box_loss(box_predictions, box_target, box_mask)
				loss = loss1 + loss2
				loss.backward()
			trainer.step(batch_size)
			cls_metric.update([cls_target], [nd.transpose(class_predictions, (0,2,1))])
			box_metric.update([box_target], [box_predictions * box_mask])
			if (i + 1) % log_interval == 0:
				name1, val1 = cls_metric.get()
				name2, val2 = box_metric.get()
				print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f'%(epoch ,i, batch_size/(time.time()-btic), name1, val1, name2, val2))
		name1, val1 = cls_metric.get()
		name2, val2 = box_metric.get()
		print('[Epoch %d] training: %s=%f, %s=%f'%(epoch, name1, val1, name2, val2))
		print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
		if epoch%5==0:
			net.save_params('models/ssd_%d.params' % epoch)
if __name__ == '__main__':
	train()