from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt


img = Image.open("cat.jpg")

enhancer = ImageEnhance.Brightness(img)
img_bright = enhancer.enhance(1.5)

plt.subplot(1,2,1)
plt.imshow(img)
plt.axis("off")
plt.title("Originale")

plt.subplot(1,2,2)
plt.imshow(img_bright)
plt.axis("off")
plt.title("Plus lumineuse")

img_bright.save("results/image_luminosite.png")
plt.show()
