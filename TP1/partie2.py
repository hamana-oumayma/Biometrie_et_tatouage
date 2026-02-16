from PIL import Image
import matplotlib.pyplot as plt
import os


# charger image
img = Image.open("cat.jpg")

# redimensionnement
img_resized = img.resize((200, 200))

# affichage
plt.subplot(1,2,1)
plt.imshow(img)
plt.axis("off")
plt.title("Originale")

plt.subplot(1,2,2)
plt.imshow(img_resized)
plt.axis("off")
plt.title("Redimensionnée")

# sauvegarde
img_resized.save("results/image_redimensionnee.png")

plt.show()
