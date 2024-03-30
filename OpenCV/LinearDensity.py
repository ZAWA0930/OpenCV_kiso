#線形濃度変換

import cv2
import numpy as np

#入力画像読み込み
img=cv2.imread("data/src/Berry.jpg")

#グレースケールに変換
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

#線形濃度変換
a,k=0.7,20

zmin,zmax=20.0,220.0

#変換1(画素値をa倍)
#gray=a*gray

#変換2(コントラストを全体的に明るく・暗く)
#gray=gray+k

#変換3(コントラストの強弱)
gray = a * (gray - 127.0) + 127.0

# 変換4(ヒストグラムの拡張（伸張）)
#gray = gray.max() * (gray - zmin)/(zmax - zmin) 

# 画素値を0～255の範囲内に収める
gray[gray < 0] = 0
gray[gray > 255] = 255

# 結果の出力
cv2.imwrite("output/LD4.jpg", gray)