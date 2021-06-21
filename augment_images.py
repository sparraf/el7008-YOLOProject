import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import pandas as pd

from tqdm import tqdm
import time

millis = np.uint16(round(time.time() * 1000))

ia.seed(millis)

# Directorio genérico donde se encuentran guardados los ejemplos originales (no augmentation)
IMG_DIR = './examples/example_{}.png'
# Directorio genérico donde se encuentran guardados los archivos .txt de cada ejemplo original
BBOX_DIR = './examples/example_{}.txt'

# Directorio genérico donde se guardarán las imagenes generadas por augmentation
AUG_IMG_DIR = './aug/aug_example_{}.png'
# Directorio genérico donde se guardarán los archivos .txt de cada imagen generada
AUG_BBOX_DIR = './aug/aug_example_{}.txt'

# Numero de ejemplos originales que existen
N_EXAMPLES = 1500

class YOLOBbox:
    
    def __init__(self, yolo_class=None, center_x=None, center_y=None, width=None, height=None, img_width=None,
                 img_height=None):
        # YOLO bbox coordinates
        self.yolo_class = int(yolo_class)
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        
        # Source image dimensions
        self.img_width = img_width
        self.img_height = img_height
        
        # imgaug bbox coordinates
        self.x1 = (center_x - width/2) * img_width
        self.x2 = (center_x + width/2) * img_width
        self.y1 = (center_y - height/2) * img_height
        self.y2 = (center_y + height/2) * img_height
        
    
    def get_YOLO_bbox(self):
        return [self.yolo_class, self.center_x, self.center_y, self.width, self.height]
    
    
    def get_imgaug_bbox(self):
        return ia.BoundingBox(self.x1, self.y1, self.x2, self.y2)
    
    
    def update(self, augmented_bbox):
        self.x1 = augmented_bbox.x1_int
        self.x2 = augmented_bbox.x2_int
        self.y1 = augmented_bbox.y1_int
        self.y2 = augmented_bbox.y2_int
        
        self.center_x = np.float32(augmented_bbox.center_x / self.img_width)
        self.center_y = np.float32(augmented_bbox.center_y / self.img_height)
        self.width = np.float32(augmented_bbox.width / self.img_width)
        self.height = np.float32(augmented_bbox.height / self.img_height)
        
        
def read_txt_bboxes(bbox_path, img_width, img_height):
    bboxes = []
    bboxes_df = pd.read_csv(bbox_path, delim_whitespace=True, header=None,
                            names=['class', 'x', 'y', 'width', 'height'])
    bboxes_arr = bboxes_df.values
    for bbox in bboxes_arr:
        yolo_class = bbox[0]
        x = bbox[1]
        y = bbox[2]
        width = bbox[3]
        height = bbox[4]
        bboxes.append(YOLOBbox(yolo_class, x, y, width, height, img_width, img_height))
    return bboxes


def read_img_and_bboxes(img_path, bbox_path):
    img = cv2.imread(img_path)
    bboxes = read_txt_bboxes(bbox_path, img.shape[1], img.shape[0])
    return img, bboxes

# Create augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Multiply((0.6, 1.2), per_channel=0.0),
    iaa.ContrastNormalization((0.5, 1.25)),
    iaa.Grayscale(alpha=(0.0, 0.5)),
    iaa.Sometimes(0.4,
                 iaa.Sharpen(alpha=(0, 0.65), lightness=(0.75, 1.15))
                 ),
    iaa.Sometimes(0.5,
                 iaa.GaussianBlur(sigma=(0, 1.0))
                 ),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.0),
    iaa.Sometimes(0.5,
                 iaa.Affine(
                     scale={"x": (0.5, 1.2), "y": (0.5, 1.2)},
                     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                     rotate=(-30, 30),
                 ))
], random_order=True)

# Augment images
for k in tqdm(range(N_EXAMPLES*5)):
    i = k % N_EXAMPLES
    img_path = IMG_DIR.format(i)
    bbox_path = BBOX_DIR.format(i)
    
    img, bboxes = read_img_and_bboxes(img_path, bbox_path)
    bbs = ia.BoundingBoxesOnImage([
        bbox.get_imgaug_bbox() for bbox in bboxes
    ], shape=img.shape)
    
    seq_det = seq.to_deterministic()
    
    img_aug = seq_det.augment_images([img])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0].remove_out_of_image().cut_out_of_image()
    
    aug_bboxes = []
    for j in range(len(bbs_aug.bounding_boxes)):
        bboxes[j].update(bbs_aug.bounding_boxes[j])
        aug_bboxes.append(bboxes[j])
    
    new_bboxes = []
    for bbox in aug_bboxes:
        new_bboxes.append(bbox.get_YOLO_bbox())
    new_bboxes = np.array(new_bboxes)

    new_bboxes_df = pd.DataFrame(new_bboxes, dtype=np.float32)
    new_bboxes_df[0] = pd.to_numeric(new_bboxes_df[0], downcast='integer')

    # Write augmented image and bounding boxes
    cv2.imwrite(AUG_IMG_DIR.format(k), img_aug)
    new_bboxes_df.to_csv(AUG_BBOX_DIR.format(k), sep=' ', header=False, index=False)

