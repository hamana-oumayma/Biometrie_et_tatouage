from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

img = Image.open("cat.jpg")
img_gray = img.convert("L")

img_edges = img_gray.filter(ImageFilter.FIND_EDGES)

plt.subplot(1,2,1)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_edges, cmap="gray")
plt.axis("off")

img_edges.save("results/image_contours.png")
plt.show()
