import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import math

# img = cv2.imread('input/colorImage.jpg')
img=cv2.imread("data/src/cone4.jpg")

#画像のサイズを小さくする（前処理）
height = img.shape[0]
width = img.shape[1]
resized_img = cv2.resize(img,(round(width/4), round(height/4)))

# HSVに変換
resized_img_HSV = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", resized_img_HSV)

###初期値
# low_hsv_min = np.array([0, 200,50])
# low_hsv_max = np.array([1, 255, 255])

low_hsv_min = np.array([0, 200,50])
low_hsv_max = np.array([1, 255, 255])

#画像の2値化（Hueが0近辺）
maskHSV_low = cv2.inRange(resized_img_HSV,low_hsv_min,low_hsv_max)



##初期値
# high_hsv_min = np.array([179, 200,0])
# high_hsv_max = np.array([179, 255, 255])

high_hsv_min = np.array([0, 164,172])
high_hsv_max = np.array([255, 255, 255])

#画像の2値化（Hueが179近辺）
maskHSV_high = cv2.inRange(resized_img_HSV,high_hsv_min,high_hsv_max)

#２つの領域を統合
hsv_mask = maskHSV_low | maskHSV_high

#画像のマスク（合成）
resultHSV = cv2.bitwise_and(resized_img, resized_img, mask = hsv_mask)

cv2.imshow("Result HSV", resultHSV)
cv2.imshow("Result mask", hsv_mask)

# h,w=hsv_mask.shape[:2]
# #輪郭抽出
# contours, hierarchy = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# #塗りつぶし画像の作成
# black_img=np.zeros((h,w),np.uint8)
# cv2.drawContours(black_img, contours, 0, 255, -1)
# cv2.imwrite('img_thresh2.jpg',black_img)
#重心計算
M = cv2.moments(hsv_mask, False)
x,y= int(M["m10"]/M["m00"]) , int(M["m01"]/M["m00"])
print('mom=('+str(x)+','+str(y)+')')
cv2.circle(img, (x,y), 10, 255, -1)

# x, y = round(x), round(y)
#     # 重心位置に x印を書く
# cv2.line(hsv_mask, (x-5,y-5), (x+5,y+5), (0, 0, 255), 2)
# cv2.line(hsv_mask, (x+5,y-5), (x-5,y+5), (0, 0, 255), 2)


# cv2.imshow("Image", hsv_mask)

a,b=hsv_mask.shape

# print(hsv_mask.shape)


# ラジアン単位を取得
radian = math.atan2(b/2 - y, a/2 - x)
print(radian)
# 0.7853981633974483

# ラジアン単位から角度を取得
degree = radian * (180 / math.pi)
print(degree)
cv2.waitKey(0)
cv2.destroyAllWindows()