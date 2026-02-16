from PIL import Image
import matplotlib.pyplot as plt


img = Image.open("cat.jpg")

img_gray = img.convert("L")

plt.subplot(1,2,1)
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")

img_gray.save("results/image_gris.png")
plt.show()
