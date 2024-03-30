#ガンマ補正
#https://algorithm.joho.info/image-processing/gamma-correction/

import cv2
import numpy as np

# #入力画像読み込み
# img=cv2.imread("data/src/Berry.jpg")

# #グレースケールに変換
# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


# #ガンマ補正用パラメータ
# gamma=0.5

# #画素値の最大値
# Imax=gray.max()

# #ガンマ補正
# gray=Imax*(gray/Imax)**(1/gamma)

# #出力
# cv2.imwrite("output/gamma.jpg",gray)

#=====LUT(ルックアップテーブル)=====
#画像の濃度変換をおこなうときの計算を省略することで処理を高速化する手法

# 入力画像を読み込み
img=cv2.imread("data/src/Berry.jpg")

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ガンマ補正の調整用パラメータ
gamma = 0.5

# 画素値の最大値
imax = gray.max()

# ガンマ補正用のルックアップテーブルを作成
lookup_table = np.zeros((256, 1), dtype='uint8')

for i in range(256):
	lookup_table[i][0] = imax * pow(float(i) / imax, 1.0 / gamma)

# ルックアップテーブルで計算
gray_gamma = cv2.LUT(gray, lookup_table)

# 結果の出力
cv2.imwrite("output/gammaLUT.jpg",gray_gamma)