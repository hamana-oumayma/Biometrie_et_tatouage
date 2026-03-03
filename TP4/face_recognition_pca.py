import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class FaceRecognitionPCA:

    def __init__(self, n_components=30):
        """
        Initialise :
        - détecteur Viola-Jones
        - nombre de composantes principales
        - variables internes
        """
        self.n_components = n_components
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.mean = None
        self.eigenvectors = None
        self.projections = None
        self.labels = None

    

    def detect_face(self, image):
        """
        Détection et extraction visage 100x100
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )

        if len(faces) == 0:
            return None

        
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))

        return face



    def load_dataset(self, dataset_path):
        X = []
        y = []

        for label in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, label)

            if not os.path.isdir(person_path):
                continue

            for file in os.listdir(person_path):
                img_path = os.path.join(person_path, file)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                face = self.detect_face(image)
                if face is not None:
                    X.append(face.flatten())
                    y.append(label)

        return np.array(X), np.array(y)

    

    def compute_pca(self, X):

        
        self.mean = np.mean(X, axis=0)

        
        X_centered = X - self.mean

        
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

       
        eigenvectors = Vt.T
       
        self.eigenvectors = eigenvectors[:, :self.n_components]

       
        self.projections = np.dot(X_centered, self.eigenvectors)

    

    def project(self, face_vector):
        face_centered = face_vector - self.mean
        return np.dot(face_centered, self.eigenvectors)

   

    def recognize(self, image_path, threshold=3000):

        image = cv2.imread(image_path)
        face = self.detect_face(image)

        if face is None:
            return None, None, "No Face Detected"

        face_vector = face.flatten()
        projection = self.project(face_vector)

        distances = np.linalg.norm(self.projections - projection, axis=1)

        min_dist = np.min(distances)
        index = np.argmin(distances)
        identity = self.labels[index]

        decision = "Match" if min_dist < threshold else "No Match"

        return identity, min_dist, decision
    

# PROGRAMME PRINCIPAL 


if __name__ == "__main__":

    dataset_path = "dataset"
    test_image = "test.jpg"

    model = FaceRecognitionPCA(n_components=30)

    print("Chargement base...")
    X, y = model.load_dataset(dataset_path)
    print(f"Dataset chargé: {len(X)} images, {len(set(y))} personnes", flush=True)
    model.labels = y

    print("Calcul PCA...", flush=True)
    try:
        model.compute_pca(X)
        print("PCA Calculé. Nombre de composantes:", model.eigenvectors.shape, flush=True)
    except Exception as e:
        print(f"Erreur PCA: {e}", flush=True)
        import traceback
        traceback.print_exc()

    print("Début reconnaissance...")
    try:
        identity, distance, decision = model.recognize(test_image, threshold=3000)
        print("Reconnaissance complétée")
        print("Distance minimale :", distance)
        print("Identité prédite :", identity)
        print("Décision :", decision)
    except Exception as e:
        print(f"Erreur : {e}")
        import traceback
        traceback.print_exc()