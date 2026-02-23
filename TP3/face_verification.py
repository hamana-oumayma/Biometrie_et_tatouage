import cv2
import numpy as np
from scipy.spatial.distance import euclidean


class FaceVerificationSystem:

    def __init__(self):
        
        self.face_cascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml"
        )

        self.reference_features = None

    
    # Détection de visage
    
    def detect_face(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None, None

        
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest

        face = gray[y:y+h, x:x+w]

        return face, (x, y, w, h)

  
    # Extraction LBP
   
    def extract_lbp_features(self, face_image):

        face_image = cv2.resize(face_image, (128, 128))
        lbp = np.zeros_like(face_image)

        for i in range(1, face_image.shape[0]-1):
            for j in range(1, face_image.shape[1]-1):

                center = face_image[i, j]
                binary_string = ""

                neighbors = [
                    face_image[i-1, j-1],
                    face_image[i-1, j],
                    face_image[i-1, j+1],
                    face_image[i, j+1],
                    face_image[i+1, j+1],
                    face_image[i+1, j],
                    face_image[i+1, j-1],
                    face_image[i, j-1]
                ]

                for neighbor in neighbors:
                    if neighbor >= center:
                        binary_string += "1"
                    else:
                        binary_string += "0"

                lbp[i, j] = int(binary_string, 2)

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=256,
            range=(0, 256)
        )

        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        return hist

   
    # Setup image référence
  
    def setup_reference(self, image_path):

        image = cv2.imread(image_path)

        face, coords = self.detect_face(image)

        if face is None:
            print("Aucun visage détecté dans l'image référence")
            return

        self.reference_features = self.extract_lbp_features(face)

        print("Visage de référence enregistré.")

   
    # Vérification visage
  
    def verify_face(self, image_path, threshold=0.75):

        image = cv2.imread(image_path)

        face, coords = self.detect_face(image)

        if face is None:
            print("Aucun visage détecté")
            return

        features = self.extract_lbp_features(face)

        distance = euclidean(self.reference_features, features)
        similarity = 1 - distance

        if similarity >= threshold:
            result = "MATCH"
            color = (0, 255, 0)
        else:
            result = "NO MATCH"
            color = (0, 0, 255)

        x, y, w, h = coords

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            image,
            result,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        print("Similarity :", similarity)
        print("Decision :", result)

        cv2.imshow("Verification", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# PROGRAMME PRINCIPAL

if __name__ == "__main__":

    system = FaceVerificationSystem()

    reference_image = "ref.png"
    test_image = "test.png"

    system.setup_reference(reference_image)
    system.verify_face(test_image, threshold=0.75)