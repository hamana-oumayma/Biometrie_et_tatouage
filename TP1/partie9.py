from PIL import Image, ImageOps
import matplotlib.pyplot as plt

img = Image.open("cat.jpg")
img_gray = img.convert("L")

img_eq = ImageOps.equalize(img_gray)

plt.subplot(1,2,1)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_eq, cmap="gray")
plt.axis("off")

img_eq.save("results/image_egalisee.png")
plt.show()
