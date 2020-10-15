import cv2
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from retinaface import RetinaFace
from resnet56_predict import Predict

flags.DEFINE_string('weights_path', './data/retinafaceweights.npy',
                    'network weights path')
flags.DEFINE_string('sample_img', './sample-images/random_internet_selfie.jpg', 'image to test on')
flags.DEFINE_string('save_destination', 'retinaface_tf2_output.jpg', "destination image")
flags.DEFINE_float('det_thresh', 0.9, "detection threshold")
flags.DEFINE_float('nms_thresh', 0.4, "nms threshold")
flags.DEFINE_bool('use_gpu_nms', True, "whether to use gpu for nms")


def _main(_argv):
    detector = RetinaFace(FLAGS.weights_path, FLAGS.use_gpu_nms, FLAGS.nms_thresh)
    img = cv2.imread(FLAGS.sample_img)
    faces, landmarks = detector.detect(img, FLAGS.det_thresh)
    predict = Predict('data/resnet56_fer_pretrained.h5')
    w=0
    if faces is not None:
        print('found', faces.shape[0], 'faces')
        for i in range(faces.shape[0]):
            w += 1
            box = faces[i].astype(np.int)
            crop_img = img[box[1]-10:box[3]+10, box[0]-10:box[2]+10]
            if (crop_img.size != 0):
                crop_img = image_resize(crop_img, 48, 48)
                crop_img = cv2.resize(crop_img, (48, 48))
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                crop_img = np.array(crop_img.reshape([1, crop_img.shape[0], crop_img.shape[1], 1]))
                result = predict.predict_emotion(crop_img)
                print(result)



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
