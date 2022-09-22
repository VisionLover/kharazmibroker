import time
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import matplotlib.pyplot as plt
from selenium import webdriver

# image = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
# image = cv2.medianBlur(image, 9)
# _, image = cv2.threshold(image, 185, 255, cv2.THRESH_BINARY)
# cv2.imwrite("Rotation.png", image)

# img_template = cv2.imread('cropped1.png', 0)
# img = cv2.imread('cropped2.png', 0)
# print(type(img))
# orb = cv2.ORB_create()
#
# kp1, des1 = orb.detectAndCompute(img_template, None)
# kp2, des2 = orb.detectAndCompute(img, None)
#
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
#
# img_out = cv2.drawMatches(img_template, kp1, img, kp2, matches[:20000], None, flags=2)
# plt.imshow(img_out)
# plt.show()







image = cv2.imread("cropped3.png")
(height, w) = image.shape[:2]
if not False:
    center = (w / 2, height / 1.8)
image = cv2.warpAffine(
    image,
    cv2.getRotationMatrix2D(center, -80, 1.0),
    (w, height)
)
cv2.imwrite("6.png", image)
image = cv2.imread("6.png")
lower = (0, 0, 0)  # lower bound for each channel
upper = (250, 250, 250)  # upper bound for each channel

# create the mask and use it to change the colors
mask = cv2.inRange(image, lower, upper)
image[mask != 0] = [255, 255, 255]
cv2.imwrite("6.png", image)





# z = 0
# found = "no"
# while z > -180:
#     image = cv2.imread("cropped1.png")
#     (height, w) = image.shape[:2]
#     if not False:
#         center = (w / 2, height / 2)
#     image = cv2.warpAffine(
#         image,
#         cv2.getRotationMatrix2D(center, z, 1.0),
#         (w, height)
#     )
#     cv2.imwrite("Rotation.png", image)
#     image = cv2.imread("Rotation.png")
#     lower = (0, 0, 0)  # lower bound for each channel
#     upper = (250, 250, 250)  # upper bound for each channel
#
#     # create the mask and use it to change the colors
#     mask = cv2.inRange(image, lower, upper)
#     image[mask != 0] = [255, 255, 255]
#     cv2.imwrite("Rotation.png", image)
#
#     imageA = cv2.imread('5.png')
#     imageB = cv2.imread('Rotation.png')
#     grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#     grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
#     (score, diff) = ssim(grayA, grayB, full=True)
#     diff = (diff * 255).astype("uint8")
#     print(score)
#     if score > 0.85:
#         print("true")
#         print("SSIM: {}".format(score))
#         found = "yes"
#         break
#     else:
#         print("false")
#     z = z - 1




# image = cv2.imread("cropped1.png", cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
# image = cv2.medianBlur(image, 9)
# _, image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
# cv2.imwrite("cropped1.png", image)
#
#
#
# final = []
# imageA = cv2.imread('cropped4.png')
# images = [cv2.imread('1.png'),cv2.imread('2.png'),cv2.imread('3.png'),cv2.imread('4.png'),cv2.imread('5.png'),cv2.imread('6.png'),cv2.imread('8.png'),cv2.imread('9.png')]
#
# # convert the images to grayscale
# grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
# y = 0
# grays = []
# while y < 8:
#     grays.append(cv2.cvtColor(images[y], cv2.COLOR_BGR2GRAY))
#     y = y + 1
# x = 0
# while x < 360:
#     image = cv2.imread("cropped1.png")
#     (height, w) = image.shape[:2]
#     if not False:
#         center = (w / 2, height / 2)
#     image = cv2.warpAffine(
#         image,
#         cv2.getRotationMatrix2D(center, x, 1.0),
#         (w, height)
#     )
#     cv2.imwrite("Rotation.png", image)
#
#     imageA = cv2.imread('Rotation.png')
#     grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#     (score, diff) = ssim(grayA, grays[6], full=True)
#     diff = (diff * 255).astype("uint8")
#
#     if score > 0.9:
#         print("true", x)
#         print("SSIM: {}".format(score))
#         final.append(x)
#         found = "found"
#     else:
#         print("false", x)
#     x = x + 1


# z = 0
# while z < 360:
#
#     image = cv2.imread("cropped4.png")
#     (height, w) = image.shape[:2]
#     if not False:
#         center = (w / 2, height / 2)
#     image = cv2.warpAffine(
#         image,
#         cv2.getRotationMatrix2D(center, z, 1.0),
#         (w, height)
#     )
#     cv2.imwrite("Rotation.png", image)
#
#     final = []
#     imageA = cv2.imread('Rotation.png')
#     images = [cv2.imread('1.png'),cv2.imread('2.png'),cv2.imread('3.png'),cv2.imread('4.png'),cv2.imread('5.png'),cv2.imread('6.png'),cv2.imread('8.png'),cv2.imread('9.png')]
#
#     # convert the images to grayscale
#     grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#     y = 0
#     grays = []
#     while y < 8:
#         grays.append(cv2.cvtColor(images[y], cv2.COLOR_BGR2GRAY))
#         y = y + 1
#     x = 0
#     while x < 8:
#         imageA = cv2.imread('Rotation.png')
#         grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#         (score, diff) = ssim(grayA, grays[x], full=True)
#         diff = (diff * 255).astype("uint8")
#
#         if score > 0.9:
#             print("true", x)
#             print("SSIM: {}".format(score))
#             final.append(x)
#             found = "found"
#             break
#         else:
#             print("false", x)
#             print(z)
#         x = x + 1
#
#     z = z + 1