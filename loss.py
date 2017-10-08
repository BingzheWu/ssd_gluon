from models import ToySSD
import mxnet 
from mxnet.gluon import nn
import mxnet.gluon  as gluon
import time
from mxnet.contrib.ndarray import MultiBoxTarget

class FocalLoss(gluon.loss.Loss):
	def __init__(self, axis = -1, alpha = 0.25, gamma = 2, batch_axis = 0, **kwargs):
		super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
		self._axis = axis
		self._alpha = alpha
		self._gamma = gamma
	def hybrid_forward(self, F, output, label):
		output = F.softmax(output)
		pt = F.pick(output, label, axis = self._axis, keepdims = True)
		loss = -self._alpha*((1-pt)**self._gamma)*F.log(pt)
		return F.mean(loss, axis = self._batch_axis, exclude = True)
class SmoothL1Loss(gluon.loss.Loss):
	def __init__(self, batch_axis = 0, **kwargs):
		super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)
	def hybrid_forward(self, F, output, label, mask):
		loss = F.smooth_l1((output-label)*mask, scalar = 1.0)
		return F.mean(loss, self._batch_axis, exclude = True)

