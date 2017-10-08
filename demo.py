import numpy as np
import cv2
from models import ToySSD
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxDetection
import mxnet as mx
import time
def preprocess(image):
    """Takes an image and apply preprocess"""
    data_shape = 300
    image = cv2.resize(image, (data_shape, data_shape))
    image = image[:, :, (2, 1, 0)]
    image = image.astype(np.float32)
    #image -= np.array([128, 128, 128])
    image = image /255.0
    image = np.transpose(image, (2, 0, 1))
    image = image[np.newaxis, :]
    image = nd.array(image)
    return image
def inference(x, epochs= 295):
    ctx = mx.cpu(1)
    net = ToySSD(1)
    start_time = time.time()
    net.load_params('models/ssd_%d.params' % epochs, ctx)
    anchors, cls_preds, box_preds = net(x.as_in_context(ctx))
    cls_probs = nd.SoftmaxActivation(nd.transpose(cls_preds, (0,2,1)), mode = 'channel')
    output = MultiBoxDetection(*[cls_probs, box_preds, anchors], force_suppress = False, clip = False, nms_threshold = 0.001 )
    end_time = time.time()
    print(end_time-start_time)
    return output
def display(img, out, thresh = 0.0):
    import random
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.figsize'] = (10,10)
    pens = dict()
    plt.clf()
    plt.imshow(img)
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        if cid not in pens:
            pens[cid] = (random.random(), random.random(), random.random())
        scales = [img.shape[1], img.shape[0]]*2
        xmin, ymin, xmax, ymax = [int(p*s) for p,s in zip(det[2:6].tolist(), scales)]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill = False,
            edgecolor = pens[cid], linewidth =3 )
        plt.gca().add_patch(rect)
        text = 'face'
        plt.gca().text(xmin, ymin-2, '{:s} {:.3f}'.format(text, score),
                       bbox=dict(facecolor=pens[cid], alpha=0.5),
                       fontsize=12, color='white')
    plt.show()
def demo(img_path):
    image = cv2.imread(img_path)
    img = preprocess(image)
    out = inference(img)
    display(image[:,:,(2,1,0)], out[0].asnumpy(), thresh = 0.40)
if __name__ == '__main__':
    import sys
    args = sys.argv
    image_path = args[1]
    demo(image_path)
