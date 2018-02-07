from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_preprocess_input
from tqdm import tqdm
import cv2
import numpy as np

from .extract_bottleneck_features import *

dog_names = [
    "Affenpinscher",
    "Afghan_hound",
    "Airedale terrier",
    "Akita",
    "Alaskan malamute",
    "American eskimo dog",
    "American foxhound",
    "American staffordshire terrier",
    "American water spaniel",
    "Anatolian shepherd dog",
    "Australian cattle dog",
    "Australian shepherd",
    "Australian terrier",
    "Basenji",
    "Basset hound",
    "Beagle",
    "Bearded collie",
    "Beauceron",
    "Bedlington terrier",
    "Belgian malinois",
    "Belgian sheepdog",
    "Belgian tervuren",
    "Bernese mountain dog",
    "Bichon frise",
    "Black and tan coonhound",
    "Black russian terrier",
    "Bloodhound",
    "Bluetick coonhound",
    "Border collie",
    "Border terrier",
    "Borzoi",
    "Boston terrier",
    "Bouvier des flandres",
    "Boxer",
    "Boykin spaniel",
    "Briard",
    "Brittany",
    "Brussels griffon",
    "Bull terrier",
    "Bulldog",
    "Bullmastiff",
    "Cairn terrier",
    "Canaan dog",
    "Cane corso",
    "Cardigan welsh corgi",
    "Cavalier king charles spaniel",
    "Chesapeake bay retriever",
    "Chihuahua",
    "Chinese crested",
    "Chinese shar-pei",
    "Chow chow",
    "Clumber spaniel",
    "Cocker spaniel",
    "Collie",
    "Curly-coated retriever",
    "Dachshund",
    "Dalmatian",
    "Dandie dinmont terrier",
    "Doberman pinscher",
    "Dogue de bordeaux",
    "English cocker spaniel",
    "English setter",
    "English springer spaniel",
    "English toy spaniel",
    "Entlebucher mountain dog",
    "Field spaniel",
    "Finnish spitz",
    "Flat-coated retriever",
    "French bulldog",
    "German pinscher",
    "German shepherd dog",
    "German shorthaired pointer",
    "German wirehaired pointer",
    "Giant schnauzer",
    "Glen of imaal terrier",
    "Golden retriever",
    "Gordon setter",
    "Great dane",
    "Great pyrenees",
    "Greater swiss mountain dog",
    "Greyhound",
    "Havanese",
    "Ibizan hound",
    "Icelandic sheepdog",
    "Irish red and white setter",
    "Irish setter",
    "Irish terrier",
    "Irish water spaniel",
    "Irish wolfhound",
    "Italian greyhound",
    "Japanese chin",
    "Keeshond",
    "Kerry blue terrier",
    "Komondor",
    "Kuvasz",
    "Labrador retriever",
    "Lakeland terrier",
    "Leonberger",
    "Lhasa apso",
    "Lowchen",
    "Maltese",
    "Manchester terrier",
    "Mastiff",
    "Miniature schnauzer",
    "Neapolitan mastiff",
    "Newfoundland",
    "Norfolk terrier",
    "Norwegian buhund",
    "Norwegian elkhound",
    "Norwegian lundehund",
    "Norwich terrier",
    "Nova scotia duck tolling retriever",
    "Old english sheepdog",
    "Otterhound",
    "Papillon",
    "Parson russell terrier",
    "Pekingese",
    "Pembroke welsh corgi",
    "Petit basset griffon vendeen",
    "Pharaoh hound",
    "Plott",
    "Pointer",
    "Pomeranian",
    "Poodle",
    "Portuguese water dog",
    "Saint bernard",
    "Silky terrier",
    "Smooth fox terrier",
    "Tibetan mastiff",
    "Welsh springer spaniel",
    "Wirehaired pointing griffon",
    "Xoloitzcuintli",
    "Yorkshire terrier"
]

class Brain:
    def __init__(self, weights_file):
        self.model = Sequential()
        self.model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        self.model.add(Dense(133, activation='softmax'))
        self.model.load_weights(weights_file)

        self.ResNet50_model = ResNet50(weights='imagenet')
        self.face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        self.xception = Xception(weights='imagenet', include_top=False)


    def predict_breed(self, img_path):
        bottleneck_feature = extract(self.path_to_tensor(img_path), self.xception, xception_preprocess_input)
        predicted_vector = self.model.predict(bottleneck_feature)
        return dog_names[np.argmax(predicted_vector)]


    def path_to_tensor(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)


    def paths_to_tensor(self, img_paths):
        list_of_tensors = [self.path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)


    def ResNet50_predict_labels(self, img_path):
        img = preprocess_input(self.path_to_tensor(img_path))
        return np.argmax(self.ResNet50_model.predict(img))


    def detect_dog(self, img_path):
        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151))


    def detect_face(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray)
        return len(faces) > 0


    def find_breed(self, image_path):
        is_human = self.detect_face(image_path)
        is_dog = self.detect_dog(image_path)

        output = "No humans nor dogs found..."
        breed = None

        if is_human:
            output = "This human looks like a "
            breed = self.predict_breed(image_path)
        elif is_dog:
            output = "This dog is a "
            breed = self.predict_breed(image_path)

        return output, breed
