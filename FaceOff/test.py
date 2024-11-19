import os
import re
import dlib
import face_recognition_models as frm
import numpy as np
import torch as t
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage 
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageDraw, ImageChops
import cv2
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class Attack(object):
    """
    Class used to create adversarial facial recognition attacks
    """
    def __init__(self, 
        input_img, 
        target_img, 
        seed=3, 
        pretrained='vggface2'
    ):
        # Value inits
        if (seed != None) : np.random.seed(seed)
        self.MEAN = t.tensor([0.485, 0.456, 0.406], device=device)
        self.STD = t.tensor([0.229, 0.224, 0.225], device=device)
        self.LOSS = t.tensor(0, device=device)

        # Function inits
        self.imageize = ToPILImage()
        self.tensorize = ToTensor()
        self.normalize = Normalize(mean=self.MEAN.cpu().numpy(), std=self.STD.cpu().numpy())
        self.resnet = InceptionResnetV1(pretrained=pretrained).eval().to(device)
        

        self.input_tensor = self.normalize(self.tensorize(input_img).to(device))

        self.input_emb = self.resnet(
            t.stack([
                    self.input_tensor
            ])
        )
        # Target image - normalized and with embedding created.
        self.target_emb = self.resnet(
            t.stack([
                self.normalize(
                    self.tensorize(target_img).to(device)
                )
            ])
        )
        # Adversarial embedding init
        self.adversarial_emb = None
        # Face mask init
        self.mask_tensor = self._create_mask(input_img)
        # Reference tensor used to apply mask
        self.ref = self.mask_tensor
    

    def train(self, 
        epochs = 30, 
        detect=False, 
        verbose=False
    ):
        for i in range(1, epochs + 1):
            # Create adversarial tensor by applying normalized MASK to normalized INPUT
            self.view(self.mask_tensor).show()
            adversarial_tensor = self._apply(
                self.input_tensor, 
                self.normalize(self.mask_tensor),
                self.ref)
            # Create embedding
            self.adversarial_emb = self.resnet(t.stack([adversarial_tensor]))

            # Calculate two distances - from adv to input and adv to target
            distance_to_image = self._emb_distance(self.adversarial_emb, self.input_emb)
            distance_to_target = self._emb_distance(self.adversarial_emb, self.target_emb)
            grad = t.autograd.grad(outputs=-distance_to_image + distance_to_target,inputs=self.mask_tensor)[0]
            self.mask_tensor = self.mask_tensor - 0.01*grad.sign()
            self.mask_tensor.data.clamp_(0, 1).detach()

            training_information = [f'Epoch {i}: \n   Loss            = {self.LOSS.item():.7f}']
            if verbose:
                training_information.append(f'\n   Dist. to Image  = {distance_to_image:.7f}')
                training_information.append(f'\n   Dist. to Target = {distance_to_target:.7f}')
            print(''.join(training_information))    

        # Return original adversarial tensor, the adversarial image, and the mask tensor
        adversarial_image = self.imageize(self._reverse_norm(adversarial_tensor).detach())
        return adversarial_tensor, self.mask_tensor, adversarial_image


    def view(self, 
        norm_image_tensor, 
        norm_mask_tensor=None
    ):
        if norm_mask_tensor is not None:
            combined_tensor = self._apply(norm_image_tensor, norm_mask_tensor, self.ref)
            return self.imageize(self._reverse_norm(combined_tensor).detach())
        else:
            return self.imageize(self._reverse_norm(norm_image_tensor).detach())


    def _apply(self, 
        image_tensor, 
        mask_tensor, 
        reference_tensor
    ):

        return t.where((reference_tensor == 0), image_tensor, mask_tensor).to(device)


    def _create_mask(self, face_image):
        mask = Image.new('RGB', (300,300), color=(0,0,0))
        d = ImageDraw.Draw(mask)
        area = [(31, 184), (40, 214), (58, 241), (82, 263), (110, 282), (142, 287), (173, 282), (198, 262), (220, 239), (236, 213), (243, 183), (151, 115)]
        d.polygon(area, fill=(255,255,255))
        mask_array = np.array(mask)
        mask_array = mask_array.astype(np.float32)

        for i in range(mask_array.shape[0]):
            for j in range(mask_array.shape[1]):
                for k in range(mask_array.shape[2]):
                    if mask_array[i][j][k] == 255.:
                        mask_array[i][j][k] = 0.5
                    else:
                        mask_array[i][j][k] = 0

        mask_tensor = ToTensor()(mask_array).to(device)
        mask_tensor.requires_grad_(True)

        return mask_tensor


    def _emb_distance(self, tensor_1, tensor_2):
        distance_tensor = (tensor_1 - tensor_2).norm()
        return distance_tensor


    def _reverse_norm(self, image_tensor):
        return image_tensor * self.STD[:, None, None] + self.MEAN[:, None, None]



def detect_face(image_loc):
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(frm.pose_predictor_model_location())
    image = dlib.load_rgb_image(image_loc)
    dets = detector(image, 1)

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(shape_predictor(image, detection))

    face_image = Image.fromarray(dlib.get_face_chip(image, faces[0], size=300))

    return face_image


def load_data(path_to_data):
    img_files = [f for f in os.listdir(path_to_data) if re.search(r'.*\.(jpe?g|png)', f)]
    img_files_locs = [os.path.join(path_to_data, f) for f in img_files]

    image_list = []

    for loc in img_files_locs:
        image_list.append(detect_face(loc))

    return image_list
