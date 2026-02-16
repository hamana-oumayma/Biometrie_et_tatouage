from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("cat.jpg")

plt.figure()
plt.subplot(1,1,1)
plt.imshow(img)
plt.axis("off")

img.save("results/cat_new.jpg")
plt.show()
