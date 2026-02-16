from calcul import compute_ssim, decision


image1 = "Empreinte1.jpg"
image2 = "Empreinte2.jpg"

similarity = compute_ssim(image1, image2)

print("Score SSIM :", similarity)
print("Décision :", decision(similarity))
