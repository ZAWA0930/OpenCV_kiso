#-*- coding:utf-8 -*-
import cv2 
import numpy as np


#=====空間フィルタリング=====
#入力画像の注目する画素値だけでなく、その近傍にある画素値も利用し出力画像の画素値を計算する
#畳み込み演算、マスク演算とも呼ばれる
#画像から輪郭を抽出できる


# def filter2d(src,kernel):
#     #カーネルサイズ
#     m,n=kernel.shape
    
#     #畳み込み演算をしない領域の幅
#     d=int((m-1)/2)
#     h,w=src.shape[0],src.shape[1]
    
#     #出力画像の配列
#     dst=np.zeros((h,w))
    
#     for y in range(d,h-d):
#         for x in range(d,w-d):
#             #畳み込み演算
#             dst[y][x]=np.sum(src[y-d:y+d+1,x-d:x+d+1]*kernel)
    
#     return dst


# #入力画像読み込み
# img=cv2.imread("data/src/Berry.jpg")

# #グレースケールに変換
# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

# #カーネル(水平方向の輪郭検出)
# kernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

# #畳み込み演算
# dst=filter2d(gray,kernel)


# cv2.imwrite("output/filter1.jpg",dst)
            
# #----OpenCVで実装----

# #畳み込み演算
# dst=cv2.filter2D(gray,-1,kernel)


# cv2.imwrite("output/filter2.jpg",dst)
            

# #=====平均値フィルタ=====
# #注目画素の近傍の画素値の平均値を計算し、その値を新しい画素値とする
# #ノイズ除去によく使われる

# # 入力画像をグレースケールで読み込み
# gray = cv2.imread("data/src/Berry.jpg",0)

# # kernel of blur filter
# # カーネル（縦方向の輪郭検出用）
# kernel = np.array([[1/9, 1/9, 1/9],
#                    [1/9, 1/9, 1/9],
#                    [1/9, 1/9, 1/9]])

# # Spatial filtering
# # 方法2(OpenCVで実装)
# dst = cv2.filter2D(gray, -1, kernel)
# #Blurメソッド使用
# dst2 = cv2.blur(gray, ksize=(3, 3))
# # output
# # 結果を出力
# cv2.imwrite("output/filterAve1.png", dst)
# cv2.imwrite("output/filterAve2.png", dst2)




#=====ガウシアンフィルタ=====
#輪郭検出や特徴点検出の前処理などに使用
#注目画素からの距離に応じて近傍の画素値に重みをかける

# 入力画像をグレースケールで読み込み
gray = cv2.imread("data/src/Berry.jpg",0)



# kernel of gaussian
# カーネル
kernel = np.array([[1/16, 1/8, 1/16],
                   [1/8, 1/4, 1/8],
                   [1/16, 1/8, 1/16]])

# 方法1
dst = cv2.filter2D(gray, -1, kernel)

# 方法3
dst2 = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=1.3)

# 結果を出力
cv2.imwrite("output/filterG1.png", dst)
cv2.imwrite("output/filterG2.png", dst2)