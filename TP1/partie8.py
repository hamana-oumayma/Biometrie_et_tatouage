from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("cat.jpg")
img_gray = img.convert("L")

hist = img_gray.histogram()

plt.plot(hist)
plt.title("Histogramme des niveaux de gris")
plt.show()
