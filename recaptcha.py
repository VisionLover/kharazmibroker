import time
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from selenium import webdriver

from selenium.webdriver.chrome.options import Options
driver = webdriver.Chrome(r"./chromedriver")

# chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--window-size=1920x1080")
# chrome_options.add_argument('--no-sandbox')
# driver = webdriver.Chrome(options=chrome_options)
from selenium.webdriver.common.keys import Keys

driver.get("https://silver.kharazmibroker.ir/Account/Login")
time.sleep(15)
element = driver.find_element_by_id('captcha-img-plus')
element_png = element.screenshot_as_png
with open("fullpage.png", "wb") as file:
    file.write(element_png)
img = cv2.imread("fullpage.png")
image = cv2.imread("fullpage.png", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
image = cv2.medianBlur(image, 9)
_, image = cv2.threshold(image, 185, 255, cv2.THRESH_BINARY)
cv2.imwrite("Captcha (2).png", image)
# img = cv2.medianBlur(img, 5)
# e = cv2.Canny(img, 100, 200)
# cv2.imwrite("Captcha (2).png", e)
image = cv2.imread("Captcha (2).png")
h, w, ch_numbers = np.shape(image)
colorlist = []
for px in range(0, w):
    a = "yes"
    for py in range(0, h):
        print(format(image[py][px]))
        if (format(image[py][px]) == "[0 0 0]"):
            a = "no"
            break
    if a == "yes":
        colorlist.append(px)
length = len(colorlist)
finallist = []
x = 1
while x < length:
    if colorlist[x] - colorlist[x - 1] != 1:
        finallist.append(colorlist[x])
        finallist.append(colorlist[x - 1])
    x = x + 1

print(finallist)
print(h)
print(w)

# *****************************************************************

first_width = finallist[1] + 1
second_width = finallist[0]
crop_image1 = image[0:144, first_width:second_width]
cv2.imwrite("cropped1.png", crop_image1)

first_width = finallist[3] + 1
second_width = finallist[2]
crop_image2 = image[0:144, first_width:second_width]
cv2.imwrite("cropped2.png", crop_image2)

first_width = finallist[5] + 1
second_width = finallist[4]
crop_image3 = image[0:144, first_width:second_width]
cv2.imwrite("cropped3.png", crop_image3)

first_width = finallist[7] + 1
second_width = finallist[6]
crop_image4 = image[0:144, first_width:second_width]
cv2.imwrite("cropped4.png", crop_image4)

# *************************************************************************
padding = ['cropped1.png', 'cropped2.png', 'cropped3.png', 'cropped4.png']
# *************************************************************************

M = 0
while M < 4:
    image = cv2.imread(padding[M])
    lower = (0, 0, 0)  # lower bound for each channel
    upper = (0, 0, 0)  # upper bound for each channel
    # create the mask and use it to change the colors
    mask = cv2.inRange(image, lower, upper)
    image[mask != 0] = [0, 0, 255]
    cv2.imwrite(padding[M], image)
    M = M + 1

fasele = 0
while fasele < 4:
    im = cv2.imread(padding[fasele])
    row, col = im.shape[:2]
    bottom = im[row - 2:row, 0:col]
    mean = cv2.mean(bottom)[0]
    image = cv2.imread(padding[fasele])
    h, w, ch_numbers = np.shape(image)
    bordersize = (50 - w) / 2
    print(bordersize)
    if bordersize * 2 % 2 == 0:
        border = cv2.copyMakeBorder(
            im,
            top=10,
            bottom=10,
            left=(150 - w) // 2,
            right=(150 - w) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean]
        )
        cv2.imwrite(padding[fasele], border)
    else:
        border = cv2.copyMakeBorder(
            im,
            top=10,
            bottom=10,
            left=((150 - w) // 2) + 1,
            right=(150 - w) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean]
        )
        cv2.imwrite(padding[fasele], border)
    fasele = fasele + 1

grays = []
imageA = cv2.imread('cropped1.png')
images = [cv2.imread('0.png'), cv2.imread('2.png'), cv2.imread('3.png'), cv2.imread('4.png'), cv2.imread('5.png'),
          cv2.imread('6.png'), cv2.imread('8.png'), cv2.imread('9.png')]
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
y = 0
answer = 0
while y < 8:
    grays.append(cv2.cvtColor(images[y], cv2.COLOR_BGR2GRAY))
    y = y + 1

numbers = 0
final = []
while numbers < 4:
    x = 0
    maximum = 0
    while x < 8:
        z = 0
        found = "no"
        while z < 150:
            image = images[x]
            (height, w) = image.shape[:2]
            if not False:
                center = (w / 2, height / 1.8)
            image = cv2.warpAffine(
                image,
                cv2.getRotationMatrix2D(center, z, 1.0),
                (w, height)
            )
            cv2.imwrite("Rotation.png", image)
            image = cv2.imread("Rotation.png")
            lower = (0, 0, 0)  # lower bound for each channel
            upper = (250, 250, 250)  # upper bound for each channel

            # create the mask and use it to change the colors
            mask = cv2.inRange(image, lower, upper)
            image[mask != 0] = [255, 255, 255]
            cv2.imwrite("Rotation.png", image)

            imageA = cv2.imread(padding[numbers])
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            imageB = cv2.imread('Rotation.png')
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            (score, diff) = ssim(grayA, grayB, full=True)
            diff = (diff * 255).astype("uint8")
            print(score)
            if score > maximum:
                maximum = score
                answer = x
            z = z + 1
        x = x + 1

    if answer == 0:
        final.append(answer)
    if answer == 7 or answer == 6:
        final.append(answer + 2)
    else:
        final.append(answer + 1)
    numbers = numbers + 1

print("******* SUCCESS! *******")


for B in final:
    if str(B) == "1":
        final.remove(1)





time.sleep(5)
recaptcha = "".join(str(x) for x in final)
print("Captcha is : ", recaptcha)
time.sleep(10)











# search_box1 = driver.find_element_by_name('username')
# search_box2 = driver.find_element_by_name('password')
# # search_box3 = driver.find_element_by_name('capcha')
# search_box1.send_keys("mfdonline2595967")
# search_box2.send_keys("Farzad1374")
# # search_box3.send_keys(recaptcha)
# time.sleep(10)
# 
# yes = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div/div[1]/multi-login/div/div[2]/div[1]/form/div[4]/div[1]/button')
# yes.click()
# time.sleep(10)
# driver.refresh()
# time.sleep(10)
# 
# driver.find_elements_by_class_name('tp-bo-bo')[1].send_keys('مادیرا')
# time.sleep(5)
# main_sahm = driver.find_elements_by_xpath('/html/body/app-container/app-content/div/div/div/div[3]/div[1]/ul[1]/li[2]/div/symbol-search/div/angucomplete/div/div[2]/div[3]/div[1]')[0]
# main_sahm.click()
# time.sleep(3)
# 
# tedad = driver.find_element_by_xpath('//*[@id="send_order_txtCount"]')
# tedad.click()
# time.sleep(3)
# tedad.send_keys("100")
# 
# gheymat = driver.find_element_by_xpath('//*[@id="send_order_txtPrice"]')
# gheymat.click()
# time.sleep(3)
# gheymat.send_keys("63109")
# time.sleep(3)
# 
# y = 0
# while y < 10:
#     button1 = driver.find_element_by_id('send_order_btnSendOrder')
#     driver.execute_script("arguments[0].click();", button1)
#     timek = driver.find_element_by_xpath('/html/body/app-container/app-header/div/section[2]/span/clock').text
#     time.sleep(0.3)
#     print(timek)

driver.quit()
