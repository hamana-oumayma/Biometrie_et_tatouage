import os
import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


class FaceRecognitionDL:

    def __init__(self):

       
        self.detector = MTCNN()
        
        # Load pre-trained VGG16 model for feature extraction
        self.embedding_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        
        self.database = {}


    def detect_face(self, image):

        results = self.detector.detect_faces(image)

        if len(results) == 0:
            return None

        x, y, w, h = results[0]['box']

        face = image[y:y+h, x:x+w]

        face = cv2.resize(face, (160,160))

        return face


    def extract_embedding(self, face):

        face = face.astype('float32')
        
        # Resize to VGG16 input size
        face = cv2.resize(face, (224, 224))
        
        # Prepare for VGG16
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)
        
        # Extract embedding
        embedding = self.embedding_model.predict(face, verbose=0)
        
        if embedding is None or len(embedding) == 0:
            return None

        return embedding[0]


    def build_database(self, dataset_path):

        for person in os.listdir(dataset_path):

            person_path = os.path.join(dataset_path, person)

            embeddings = []

            for img_name in os.listdir(person_path):

                img_path = os.path.join(person_path, img_name)

                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                face = self.detect_face(image)

                if face is None:
                    continue

                embedding = self.extract_embedding(face)

                if embedding is None:
                    continue

                embeddings.append(embedding)

            if len(embeddings) > 0:
                self.database[person] = embeddings


    def cosine_similarity(self, emb1, emb2):

        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        return dot / (norm1 * norm2)


    def euclidean_distance(self, emb1, emb2):

        return np.linalg.norm(emb1 - emb2)


    def recognize(self, image_path, threshold=0.6):

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face = self.detect_face(image)

        if face is None:
            return None, None, "No Face"

        embedding = self.extract_embedding(face)

        if embedding is None:
            return None, None, "No Face"

        best_label = None
        best_distance = 999

        for label in self.database:

            for db_emb in self.database[label]:

                dist = self.euclidean_distance(embedding, db_emb)

                if dist < best_distance:
                    best_distance = dist
                    best_label = label

        decision = "Match" if best_distance <= threshold else "No Match"

        return best_label, best_distance, decision
    
def main():

    dataset = "dataset/"
    test_image = "test.jpeg"

    model = FaceRecognitionDL()

    print("Construction de la base...")
    model.build_database(dataset)

    print("Reconnaissance...")

    label, distance, decision = model.recognize(test_image)

    print("\nRésultat :")
    print("Identité :", label)
    print("Distance :", distance)
    print("Décision :", decision)


if __name__ == "__main__":
    main()