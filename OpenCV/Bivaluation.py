#二値化

import cv2 
import numpy as np

# #=====単純な二値化=====



# # 閾値
# threshold_value = 127

# # 入力画像
# img=cv2.imread("data/src/Berry.jpg")

# # グレースケール変換
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# #----Numpyを使用する ----

# # 出力画像用の配列生成
# threshold_img = gray.copy()

# # 方法1（NumPyで実装）
# threshold_img[gray < threshold_value] = 0 #[]の条件を満たす場合のみ置き換えを行う
# threshold_img[gray >= threshold_value] = 255

# # 結果を出力
# cv2.imwrite("output/Berry_Bivaluation1.jpg", threshold_img)

# #----OpenCVを使用----


# # 方法2（OpenCVで実装）
# ret, threshold_img2 = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY) #cv2.threshold(入力画像, 閾値, 二値化の最大値, 二値化手法)

# # 結果を出力
# cv2.imwrite("output/Berry_Bivaluation2.jpg", threshold_img2)







#=====適応的二値化処理=====

#閾値を固定せず、注目画素と周囲にある画素の画素値の平均値を閾値とする
#8近傍(注目画素＋周囲8画素の画素値)

#----Numpyを使用----

# def threshold(src,ksize=3,c=2):
#     #局所領域の幅 ex)ksize=11ならば注目画素値から5つ隣の画素値までの平均を考える
#     d=int((ksize-1)/2)
    
#     #画像の高さと幅
#     h,w=src.shape[0],src.shape[0]
    
#     #出力画像用の配列
#     dst=np.empty((h,w))
#     dst.fill(255) #すべての要素を255で埋める
    
#     #局所領域の画素数
#     N=ksize**2
    
#     for y in range(0,h):
#         for x in range(0,w):
#             #局所領域内の画素数の平均を計算し、閾値に設定
#             t=np.sum(src[y-d:y+d+1,x-d:x+d+1])/N
            
#             #求めた閾値で二値化処理
#             if (src[y][x]<t-c):
#                 dst[y][x]=0
#             else:
#                 dst[y][x]=255
#     return dst



# # 入力画像
# img=cv2.imread("data/src/Berry.jpg")

# # グレースケール変換
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# #二値化
# dst=threshold(gray,ksize=11,c=13)

# cv2.imwrite("output/Berry_Bivaluation_AD1.jpg", dst)
# #----OpenCVを使用----

# # グレースケール変換
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
# # 方法2       
# dst = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,13)

    
# # 結果を出力
# cv2.imwrite("output/Berry_Bivaluation_AD2.jpg", dst)




#=====大津の手法=====
#大津の手法（判別分析法）は、自動的に閾値を決定して二値化処理を行う手法
#ヒストグラムを最もきれいに分割できるような閾値を求める

#1  ヒストグラムを求める
#2  画素値の最大値、最小値、平均値を求める
#3  最大値と最小値の間で閾値を選ぶ
#4  閾値でヒストグラムを2つのクラスに分けます
#5  クラス1の分散、平均値、画素数を求める
#6　クラス1の分散、平均値、画素数を求める
#7　クラス内分散とクラス間分散を求める
#8  #7の分散から分離度Sを求める
#9  #3~#8を繰り返し、すべての閾値分だけ求める
#10 分離度が最大になる時の閾値を二値化処理に用いる閾値に決定



# img=cv2.imread("data/src/Berry.jpg")

# # グレースケール変換
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# #----OpenCVを使用----

# #ret, dst = cv2.threshold(入力画像, 閾値, 二値化時の最大値, 二値化方法(cv.THRESH_OTSU))
     
# ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

# # 結果を出力
# cv2.imwrite("output/Berry_Bivaluation_OTSU1.jpg", th)




# #----Numpyを使用----

# #大津の手法
# def threshold_otsu(gray,min_value=0,max_value=255):
#     #ヒストグラム算出
#     hist=[np.sum(gray==i) for i in range(256)]
    
#     s_max=(0,-10)
    
#     for th in range(256):
#         #クラス1とクラス２の画素数を計算
#         n1=sum(hist[:th])
#         n2=sum(hist[th:])
        
#         #クラス１とクラス２の画素数の平均値を計算
#         if n1==0:
#             mu1=0
#         else:
#             mu1=sum([i*hist[i] for i in range(0,th)])/n1
            
#         if n2==0:
#             mu2=0
#         else:
#             mu2=sum([i*hist[i] for i in range(th,256)])/n2
        
#         #クラス間分散の分子を計算
#         s=n1*n2*(mu1-mu2)**2
        
#         # クラス間分散の分子が最大のとき、クラス間分散の分子と閾値を記録
#         if s > s_max[1]:
#             s_max = (th, s)
    
#     #クラス間分散が最大の時の閾値を取得
#     t=s_max[0]
    
#     #算出した閾値で二値化処理
#     gray[gray < t] = min_value
#     gray[gray >= t] = max_value

#     return gray


# th2 = threshold_otsu(gray)
    
        
# cv2.imwrite("output/Berry_Bivaluation_OTSU2.jpg", th2)









#=====膨張・収縮処理してノイズを除去=====


# img=cv2.imread("data/src/Berry.jpg")

# # グレースケール変換
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# # 二値化処理
# gray[gray<127] = 0
# gray[gray>=127] = 255

# #----Numpyで実装----

# #膨張処理
# #注目画素の周りに一つでも白があれば注目画素を白くする
# def dilate(src,ksize=3):
#     #入力画面のサイズを取得
#     h,w=src.shape
#     #入力画像をコピーして出力画像配列を生成
#     dst=src.copy()
#     #注目領域の幅
#     d=int((ksize-1)/2)
    
#     for y in range(0,h):
#         for x in range(0,w):
#             #近傍に白い画素が一つでもあれば注目画素を白に塗り替える
#             roi =src[y-d:y+d+1,x-d:x+d+1] 
#             if np.count_nonzero(roi)>0: #0でない値の数>0
#                 dst[y][x]=255
                
#     return dst

# #収縮処理
# #注目画素の周りに一つでも黒があれば注目画素を黒くする

# def erode(src,ksize=3):
#     #入力画像のサイズを取得
#     h,w=src.shape
#     #入力画像をコピーして出力画像用配列を生成
#     dst=src.copy()
    
#     d=int((ksize-1)/2)
    
#     for y in range(0, h):
#         for x in range(0, w):
#             # 近傍に黒い画素が1つでもあれば、注目画素を黒色に塗り替える
#             roi = src[y-d:y+d+1, x-d:x+d+1]
#             if roi.size - np.count_nonzero(roi) > 0:
#                 dst[y][x] = 0

#     return dst



# #膨張処理、収縮処理
# dilate_img = dilate(gray, ksize=6)
# erode_img = erode(dilate_img, ksize=6)

# cv2.imwrite("output/Berry_Bivaluation_dilate1.jpg", dilate_img)
# cv2.imwrite("output/Berry_Bivaluation_erode1.jpg", erode_img)

# #----OpenCVを使用----


# # カーネルの定義
# kernel = np.ones((6, 6), np.uint8)

# # 膨張・収縮処理(方法2)
# dilate = cv2.dilate(gray, kernel)
# erode = cv2.erode(dilate, kernel)

# cv2.imwrite("output/Berry_Bivaluation_dilate2.jpg", dilate)
# cv2.imwrite("output/Berry_Bivaluation_erode2.jpg", erode)






#=====二値画像のラベリングとブロブ解析=====

img=cv2.imread("data/src/sample.png")

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#二値化    
gray= cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


# ラベリング処理
label = cv2.connectedComponentsWithStats(gray)

# ブロブ情報を項目別に抽出
n = label[0] - 1
data = np.delete(label[2], 0, 0)
center = np.delete(label[3], 0, 0)

# ラベルの個数nだけ色を用意
print("ブロブの個数:", n)
print("各ブロブの外接矩形の左上x座標", data[:,0])
print("各ブロブの外接矩形の左上y座標", data[:,1])
print("各ブロブの外接矩形の幅", data[:,2])
print("各ブロブの外接矩形の高さ", data[:,3])
print("各ブロブの面積", data[:,4])
print("各ブロブの中心座標:\n",center)


