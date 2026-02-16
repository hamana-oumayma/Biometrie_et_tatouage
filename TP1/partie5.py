from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("cat.jpg")
img_gray = img.convert("L")

T = 128
img_bin = img_gray.point(lambda p: 255 if p > T else 0)

plt.subplot(1,2,1)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_bin, cmap="gray")
plt.axis("off")

img_bin.save("results/image_binarisee.png")
plt.show()
