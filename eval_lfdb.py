import numpy as np
import cv2
from models import ToySSD
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxDetection
import mxnet as mx
import os
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
    net.load_params('models/ssd_%d.params' % epochs, ctx)
    print("load sucecuss")
    anchors, cls_preds, box_preds = net(x.as_in_context(ctx))
    cls_probs = nd.SoftmaxActivation(nd.transpose(cls_preds, (0,2,1)), mode = 'channel')
    output = MultiBoxDetection(*[cls_probs, box_preds, anchors], force_suppress = True, clip = False)
    return output
def forward(img_path, net):
    ctx = mx.gpu(1)
    img_original = cv2.imread(img_path)
    img = preprocess(img_original)
    anchors, cls_preds, box_preds = net(img.as_in_context(ctx))
    cls_probs = nd.SoftmaxActivation(nd.transpose(cls_preds, (0, 2, 1)), mode = 'channel')
    output = MultiBoxDetection(*[cls_probs, box_preds, anchors], force_suppress = True, clip = True, nms_threshold = 0.01)
    return img_original, output

def demo(img_path):
    image = cv2.imread(img_path)
    img = preprocess(image)
    out = inference(img)
    display(image[:,:,(2,1,0)], out[0].asnumpy(), thresh = 0.40)
def format_rst(im, w, h, boxes):
    boxes = boxes.asnumpy()
    boxes = boxes.squeeze()
    #boxes = boxes.tolist()
    im = im.split('fddb/')[-1].strip()
    im = im.split('.jpg')[0].strip()
    im = im.split('images/')[-1].strip()
    line = [im]
    thresh = 0.3
    boxes = [x for x in boxes if x[0] >= 0 and x[1]>thresh]
    line.append(str(len(boxes)))
    #print(boxes.shape)
    for idx,box in enumerate(boxes):
        if box[0] < 0:
            continue
        scales = [w, h]*2
        xmin, ymin, xmax, ymax = [float(p*s) for p,s in zip(box[2:6].tolist(), scales)]
        line.append("%f %f %f %f %f"%(
            xmin,
            ymin,
            xmax - xmin,
            ymax - ymin,
            float(box[1])))
    return [i+'\n' for i in line]
def generate_roc(fddb_dir):
    ctx = mx.gpu(1)
    net = ToySSD(1)
    net.load_params('models/ssd_160.params', ctx)
    for fold in os.listdir(os.path.join(fddb_dir, 'FDDB-folds')):
        if 'ellip' in fold:
            continue
        f = open(os.path.join(fddb_dir, 'FDDB-folds', fold))
        fsave = open(os.path.join(fddb_dir, 'rst', "fold-%s-out.txt" % fold[-6:-4]), 'w')
        images = [os.path.join(fddb_dir, line[:-1] + '.jpg') for line in f.readlines()]
        for idx, im_path in enumerate(images):
            #im_path = os.path.join(fddb_dir, im_path)
            if not os.path.exists(im_path):
                continue
            print(im_path)
            img, out = forward(im_path, net)
            fsave.writelines(format_rst(im_path, img.shape[1], img.shape[0], out))
        fsave.close()

if __name__ == '__main__':
    generate_roc('/home/bingzhe/datasets/fddb')
