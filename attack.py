from FaceOff.test import Attack
import cv2
import numpy as np


source = cv2.imread('/home/dcd/work/adversarial_TEST/1.bmp').astype(np.float32)
target = cv2.imread('/home/dcd/work/adversarial_TEST/2.bmp').astype(np.float32)

adversarial = Attack(source, target)
adversarial_tensor, mask_tensor, img = adversarial.train(detect=True, verbose=True)


img.show()
