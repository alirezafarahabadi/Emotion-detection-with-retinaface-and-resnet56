import cv2
import numpy as np
import os
from absl import app, flags
from absl.flags import FLAGS
from retinaface import RetinaFace

flags.DEFINE_string('weights_path', './data/retinafaceweights.npy',
                    'network weights path')
flags.DEFINE_float('det_thresh', 0.9, "detection threshold")
flags.DEFINE_float('nms_thresh', 0.4, "nms threshold")
flags.DEFINE_bool('use_gpu_nms', True, "whether to use gpu for nms")


def _main(_argv):
    detector = RetinaFace(FLAGS.weights_path, FLAGS.use_gpu_nms, FLAGS.nms_thresh)
    image_root = ['data/WIDER_test/images/', 'data/WIDER_train/images/', 'data/WIDER_val/images/']
    result_save_root = 'widerface-faces/'
    for i in range(0,3):
      create_directory(os.path.join(result_save_root, image_root[i]))
    face_numbers=0
    for k in range(0,3):
      for parent, dir_names, file_names in os.walk(image_root[k]):
        for file_name in file_names:
          if not file_name.lower().endswith('jpg'):
            continue
          face_numbers = 0
          img = cv2.imread(os.path.join(parent, file_name), cv2.IMREAD_COLOR)
          faces, landmarks = detector.detect(img, FLAGS.det_thresh)
          print(faces.shape)
          if faces is not None:
              print('found', faces.shape[0], 'faces')
              for i in range(faces.shape[0]):
                  face_numbers += 1
                  box = faces[i].astype(np.int)
                  color = (0, 0, 255)
                  crop_img = img[box[1]-10:box[3]+10, box[0]-10:box[2]+10]
                  if (crop_img.size != 0):
                    crop_img = image_resize(crop_img, 48, 48)
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    if not cv2.imwrite(os.path.join(result_save_root, image_root[k], file_name.replace('.jpg','result_{}.jpg'.format(face_numbers))), crop_img):
                      raise Exception("Could not write image")



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def create_directory(path):
  try: 
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise


if __name__ == '__main__':
    try:
        app.run(_main)
    except SystemExit:
        pass
