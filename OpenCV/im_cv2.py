# import cv2
# import numpy as np
# import math
# import sys
# img = np.zeros((480,640), np.uint8)
# cv2.imshow('src', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #=====カメラ画像の読み取り=====
# cap=cv2.VideoCapture(0)  #カメラオープン

# if not cap.isOpened():
#     print('error')
#     sys.exit()
    
# #ウインドウ作成
# cv2.namedWindow('src')

# #解像度を設定
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# #解像度の確認
# ret,img_src=cap.read()
# print(img_src.shape[1],img_src.shape[0])


# while True:
#     ret,img_src = cap.read()
#     cv2.imshow('src', img_src)
#     key = cv2.waitKey(1)
#     if key==ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
    
# #=====画像の読み取り=====

# file_src='data\src\Berry.jpg'
# file_dst='data\src\Blob.png'


# img_src = cv2.imread(file_src,1)  #入力画像をカラーで読み取り
# #img_src=cv2.imread(file_src,0)       #入力画像をグレースケールで読み取り

# ## ウィンドウの表示形式の設定
# cv2.namedWindow('src')
# cv2.namedWindow('dst')

# img_dst = cv2.flip(img_src, flipCode=0) # 画像を垂直反転させる

# # 入力画像と出力画像を表示
# cv2.imshow('src', img_src)
# cv2.imshow('dst', img_dst)

# cv2.imwrite(file_dst,img_dst) #処理結果の保存
# cv2.waitKey(0) #キー入力まち
# cv2.destroyAllWindows()

# #=====画素値の読み取り、書き込み=====

# file_src='data\src\Berry.jpg'
# file_dst='data\src\Blob.png'

# img_gray=cv2.imread(file_src,0)       #入力画像をグレースケールで読み取り
# img_src = cv2.imread(file_src,1)  #入力画像をカラーで読み取り


# cv2.namedWindow('gray')
# cv2.namedWindow('src')

# x=10
# y=50
# v=255
# r=255
# g=255
# b=0

# #グレースケール
# print(img_gray[x,y])
# img_gray[y,x]=v
# print(img_gray[y,x])

# #カラー画像の場合
# print(img_src[x,y])
# img_src[y,x]=v
# print(img_src[y,x])

# cv2.imshow('gray', img_gray)
# cv2.imshow('src', img_src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##画像のRGBを入れ替えた
# file_src='data\src\Berry.jpg'
# file_dst='data\src\Blob.png'

# img_src = cv2.imread(file_src, cv2.IMREAD_COLOR)

# cv2.namedWindow('src')
# cv2.namedWindow('dst')

# # 複数色チャンネルの分割
# img_bgr = cv2.split(img_src)
# # 青→赤，緑→青，赤→緑に変更
# img_dst = cv2.merge((img_bgr[1], img_bgr[2], img_bgr[0]))

# cv2.imshow('src', img_src)  # 入力画像を表示
# cv2.imshow('dst', img_dst)  # 出力画像を表示
# cv2.imwrite(file_dst, img_dst)  # 処理結果の保存

# cv2.waitKey(0)  # キー入力待ち
# cv2.destroyAllWindows()


# ##=====ヒストグラム=====
# file_src='data\src\Berry.jpg'
# file_dst='data\src\Bus.jpg'
# file_hst='data\src\Lena.jng'


# img_src=cv2.imread(file_src,cv2.IMREAD_GRAYSCALE) #グレースケールで読み込む

# cv2.namedWindow('src')
# cv2.namedWindow('hst')

# #ヒストグラム表示、256×100,0で初期化
# img_hst=np.zeros([100,256]).astype(np.uint8) #データ型をuint8に指定
# rows,cols=img_hst.shape

# #度数分布を求める
# hdims=[256]
# hranges=[0,256]

# hist=cv2.calcHist([img_src],[0],None,hdims,hranges)

# #度数の最大値を取得

# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)

# #ヒストグラム絵画
# for i in range(0, 255):
#   v = hist[i]
#   cv2.line(img_hst, (i, rows), (i, int(rows - rows * (v / max_val))), (255, 255, 255))

# cv2.imshow('src', img_src)  # 入力画像を表示
# cv2.imshow('hst', img_hst)  # 出力画像を表示
# cv2.imwrite('filehst.jpg', img_hst)  # 処理結果の保存

# cv2.waitKey(0)  # キー入力待ち
# cv2.destroyAllWindows()

##=====トラックバー=====
#ガンマ補正…輝度ないし三刺激値を符号化および復号するために使用される非線形操作である

# def nothing(x):
#   pass

# #画像読み込み
# file_src='data\src\Berry.jpg'

# img_src=cv2.imread(file_src,1)

# #ウィンドウ作成
# cv2.namedWindow('src')
# cv2.namedWindow('dst')

# #トラックバー作成
# cv2.createTrackbar('gamma', 'dst', 1,10,nothing)

# #入力画像を表示
# cv2.imshow('src',img_src)


# while True:
#   #トラックバーの値を取得
#   gamma = cv2.getTrackbarPos('gamma', 'dst') +1.0
  
#   #ガンマ補正
#   Y=np.ones((256,1),dtype='uint8')*0
  
#   for i in range(256):
#     Y[i][0]=255*pow(float(i)/255, 1.0/gamma)
#   img_dst=cv2.LUT(img_src,Y)
  
#   #出力画像を表示
#   cv2.imshow('dst', img_dst)
#   #qキーで終了
#   key = cv2.waitKey(1) & 0xFF
#   if key==ord('q'):
#     break
  
# cv2.destroyAllWindows()

