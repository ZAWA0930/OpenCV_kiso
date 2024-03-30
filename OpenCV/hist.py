import cv2
import numpy as np
from matplotlib import pyplot as plt

##ヒストグラムと濃度変換

# #=====グレースケールのヒストグラム=====

# # 入力画像を読み込み

# img=cv2.imread("data/src/Berry.jpg")

# # グレースケール変換
# gray = cv2.cv2tColor(img, cv2.COLOR_RGB2GRAY)

# # NumPyでのヒストグラムの算出
# # hist, bins = np.histogram(gray.ravel(),256,[0,256])
# #gray.ravelは一次元配列に変換して入力するため

# # OpenCVでのヒストグラムの算出
# hist = cv2.calcHist(gray,[0],None,[256],[0,256])

# # ヒストグラムの中身表示
# print(hist)

# # グラフの作成
# plt.xlim(0, 255) #画素値なので0~255
# plt.plot(hist)
# plt.xlabel("Pixel value", fontsize=20)
# plt.ylabel("Number of pixels", fontsize=20)
# plt.grid()
# plt.show()

# #=====RGBカラー画像のヒストグラム=====

# # 入力画像を読み込み
# img=cv2.imread("data/src/road.jpg")

# img_b, img_g, img_r = img[:,:,0], img[:,:,1], img[:,:,2]  #B,G,Rと1チャンネルごとに分解

# # NumPyでのヒストグラムの算出
# # hist_r, bins = np.histogram(r.ravel(),256,[0,256])
# # hist_g, bins = np.histogram(g.ravel(),256,[0,256])
# # hist_b, bins = np.histogram(b.ravel(),256,[0,256])

# # Opencv2でのヒストグラムの算出
# hist_r = cv2.calcHist([img_r],[0],None,[256],[0,256])
# hist_g = cv2.calcHist([img_g],[0],None,[256],[0,256])
# hist_b = cv2.calcHist([img_b],[0],None,[256],[0,256])

# print('hist_r=')
# print(hist_r)

# print('hist_g=')
# print(hist_g)

# print('hist_b=')
# print(hist_b)

# # グラフの作成
# plt.xlim(0, 255)
# plt.plot(hist_r, "-r", label="Red")
# plt.plot(hist_g, "-g", label="Green")
# plt.plot(hist_b, "-b", label="Blue")
# plt.xlabel("Pixel value", fontsize=20)
# plt.ylabel("Number of pixels", fontsize=20)
# plt.legend()
# plt.grid()
# plt.show()



# #=====ヒストグラム平均化=====


#==OpenCVを使用して実装==
#平均画素数＝全画素数/階調数=(画像の幅×高さ )/256

# 入力画像を読み込み
img=cv2.imread("data/src/Berry.jpg")

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Opencv2で実装 
dst = cv2.equalizeHist(gray) #ヒストグラム平均化

# 結果の出力

cv2.imwrite("output/Berry_hist_mean.jpg", dst)

#==numpyを使用して実装==

def equalize_hist(src):
    #画像の高さ・幅を取得
    h,w =src.shape[0],src.shape[1]
    
    #全画素数
    S=w*h
    
    #画素数の最大値
    Imax=src.max()
    
    #ヒストグラム算出
    hist,bins = np.histogram(src.ravel(), 256,[0,256])
    #gsrc.ravelは一次元配列に変換して入力するため
    
    #出力画像用の配列(要素は0)
    dst=np.empty((h,w)) #空の配列を生成
    
    for y in range(0,h):
        for x in range(0,w):
            #ヒストグラム平均化の計算式
            dst[y][x]=np.sum(hist[0:src[y][x]])*(Imax/S)
    
    return dst

dst2 = equalize_hist(gray)
cv2.imwrite("output/Berry_hist_mean2.jpg", dst2)