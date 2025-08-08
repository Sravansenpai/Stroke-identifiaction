import os
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

def extract_features(data_dir):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)

    features = []
    labels = []

    for label, class_name in enumerate(['nostroke', 'stroke']):
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)

                feature = model.predict(img, verbose=0)
                features.append(feature.flatten())
                labels.append(label)

    return np.array(features), np.array(labels)
