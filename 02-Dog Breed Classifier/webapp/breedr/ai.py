from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from tqdm import tqdm
import cv2
import numpy as np

from extract_bottleneck_features import *

dog_names = [
    "Affenpinscher",
    "Afghan_hound",
    "Airedale_terrier",
    "Akita",
    "Alaskan_malamute",
    "American_eskimo_dog",
    "American_foxhound",
    "American_staffordshire_terrier",
    "American_water_spaniel",
    "Anatolian_shepherd_dog",
    "Australian_cattle_dog",
    "Australian_shepherd",
    "Australian_terrier",
    "Basenji",
    "Basset_hound",
    "Beagle",
    "Bearded_collie",
    "Beauceron",
    "Bedlington_terrier",
    "Belgian_malinois",
    "Belgian_sheepdog",
    "Belgian_tervuren",
    "Bernese_mountain_dog",
    "Bichon_frise",
    "Black_and_tan_coonhound",
    "Black_russian_terrier",
    "Bloodhound",
    "Bluetick_coonhound",
    "Border_collie",
    "Border_terrier",
    "Borzoi",
    "Boston_terrier",
    "Bouvier_des_flandres",
    "Boxer",
    "Boykin_spaniel",
    "Briard",
    "Brittany",
    "Brussels_griffon",
    "Bull_terrier",
    "Bulldog",
    "Bullmastiff",
    "Cairn_terrier",
    "Canaan_dog",
    "Cane_corso",
    "Cardigan_welsh_corgi",
    "Cavalier_king_charles_spaniel",
    "Chesapeake_bay_retriever",
    "Chihuahua",
    "Chinese_crested",
    "Chinese_shar-pei",
    "Chow_chow",
    "Clumber_spaniel",
    "Cocker_spaniel",
    "Collie",
    "Curly-coated_retriever",
    "Dachshund",
    "Dalmatian",
    "Dandie_dinmont_terrier",
    "Doberman_pinscher",
    "Dogue_de_bordeaux",
    "English_cocker_spaniel",
    "English_setter",
    "English_springer_spaniel",
    "English_toy_spaniel",
    "Entlebucher_mountain_dog",
    "Field_spaniel",
    "Finnish_spitz",
    "Flat-coated_retriever",
    "French_bulldog",
    "German_pinscher",
    "German_shepherd_dog",
    "German_shorthaired_pointer",
    "German_wirehaired_pointer",
    "Giant_schnauzer",
    "Glen_of_imaal_terrier",
    "Golden_retriever",
    "Gordon_setter",
    "Great_dane",
    "Great_pyrenees",
    "Greater_swiss_mountain_dog",
    "Greyhound",
    "Havanese",
    "Ibizan_hound",
    "Icelandic_sheepdog",
    "Irish_red_and_white_setter",
    "Irish_setter",
    "Irish_terrier",
    "Irish_water_spaniel",
    "Irish_wolfhound",
    "Italian_greyhound",
    "Japanese_chin",
    "Keeshond",
    "Kerry_blue_terrier",
    "Komondor",
    "Kuvasz",
    "Labrador_retriever",
    "Lakeland_terrier",
    "Leonberger",
    "Lhasa_apso",
    "Lowchen",
    "Maltese",
    "Manchester_terrier",
    "Mastiff",
    "Miniature_schnauzer",
    "Neapolitan_mastiff",
    "Newfoundland",
    "Norfolk_terrier",
    "Norwegian_buhund",
    "Norwegian_elkhound",
    "Norwegian_lundehund",
    "Norwich_terrier",
    "Nova_scotia_duck_tolling_retriever",
    "Old_english_sheepdog",
    "Otterhound",
    "Papillon",
    "Parson_russell_terrier",
    "Pekingese",
    "Pembroke_welsh_corgi",
    "Petit_basset_griffon_vendeen",
    "Pharaoh_hound",
    "Plott",
    "Pointer",
    "Pomeranian",
    "Poodle",
    "Portuguese_water_dog",
    "Saint_bernard",
    "Silky_terrier",
    "Smooth_fox_terrier",
    "Tibetan_mastiff",
    "Welsh_springer_spaniel",
    "Wirehaired_pointing_griffon",
    "Xoloitzcuintli",
    "Yorkshire_terrier"
]

class Brain:
    def __init__(self):
        self.model = Sequential()
        self.model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        self.model.add(Dense(133, activation='softmax'))
        self.model.load_weights('saved_models/weights.best.xception.hdf5')

        self.ResNet50_model = ResNet50(weights='imagenet')
        self.detector = face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


    def predict_breed(self, img_path):
        bottleneck_feature = extract_Xception(self.path_to_tensor(img_path))
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
        faces = self.detector.detectMultiScale(gray)
        return len(faces) > 0


    def find_breed(self, image_path):
        is_human = self.detect_face(image_path)
        is_dog = self.detect_dog(image_path)

        output = "No humans nor dogs found..."

        if is_human:
            output = "This human looks like a {}"
        elif is_dog:
            output = "This dog is a {}"

        output = output.format(self.predict_breed(image_path))

        return output

