from preprocessus import preprocess
from skimage.metrics import structural_similarity as compare_ssim

def compute_ssim(image1_path, image2_path):

    img1 = preprocess(image1_path)
    img2 = preprocess(image2_path)

    similarity = compare_ssim(img1, img2, data_range=255)

    return similarity


def decision(similarity, threshold=0.75):

    if similarity >= threshold:
        return "ACCEPTÉE"
    else:
        return "REJETÉE"
