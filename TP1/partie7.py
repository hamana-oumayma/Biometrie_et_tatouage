from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

img = Image.open("cat.jpg")
img_gray = img.convert("L")

img_blur = img_gray.filter(ImageFilter.GaussianBlur(radius=6))

plt.subplot(1,2,1)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_blur, cmap="gray")
plt.axis("off")

img_blur.save("results/image_flou.png")
plt.show()
