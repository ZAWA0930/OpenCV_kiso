import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import copy

#==============画像表示============
# img=cv2.imread("data/src/Berry.jpg")

# cv2.imshow("", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # os.mkdir("./output")
# cv2.imwrite("output/test.jpg",img)


#==============動画表示============
# cap = cv2.VideoCapture("data/movie/Cosmos.mp4")

# if cap.isOpened() == False:
#     sys.exit()

# ret,frame = cap.read()
# h,w= frame.shape[:2]
# fourcc= cv2.VideoWriter_fourcc(*"XVID")
# dst =cv2.VideoWriter("output/test@.avi", fourcc, 60.0, (w,h))
# #dst =cv2.VideoWriter("output/test.avi", video名, FPS, 解像度(w,h))
# print(h,w)

# while True:
#     ret,frame=cap.read()
#     if ret == False:
#         break
#     cv2.imshow("img", frame)
#     dst.write(frame)
#     if cv2.waitKey(30)==27:
#         #27はescキー
#         break

# cv2.destroyAllWindows()
# cap.release()

#==============ウィンドウの調整============

# img = cv2.imread("data/src/Lena.jpg")


# cv2.namedWindow("window", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow(img,(640,480))
# cv2.imshow("", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#==============リサイズ============

# img = cv2.imread("data/src/grapes.jpg")
# cv2.imshow("", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# size=(300,200)
# img_resize=cv2.resize(img,size)
# print(img_resize.shape)

# # cv2.imshow("resize", img_resize)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# img_area=cv2.resize(img,size,interpolation=cv2.INTER_AREA)
# img_linear=cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)
# cv2.imshow("", img_area)
# cv2.waitKey(0)

# cv2.destroyAllWindows()
# cv2.imshow("", img_linear)

# cv2.waitKey(0)

# cv2.destroyAllWindows()



#==============グレースケール============
# img = cv2.imread("data/src/grapes.jpg")
# cv2.imshow("", img)
# img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# print("GRAY", img_gray.shape)
# print("HSV",img_hsv.shape)

# cv2.imshow("GRAY", img_gray)
# cv2.imshow("HSV", img_hsv)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#==============ヒストグラムRGB============

# img = cv2.imread("data/src/Lena.jpg")
# cv2.imshow("", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# color_list=["blue", "green", "red"]
# for i,j in enumerate(color_list):
#     hist = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(hist,color=j)


# plt.show()

#==============ヒストグラムGRAY============
# img = cv2.imread("data/src/Lena.jpg",0)
# cv2.imshow("", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# hist2 = cv2.calcHist([img],[0],None,[256],[0,256])
# plt.plot(hist2)
# plt.show()


#==============ヒストグラム均一化============
# img = cv2.imread("data/src/Lena.jpg",0)
# cv2.imshow("", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # hist = cv2.calcHist([img],[0],None,[256],[0,256])
# # plt.plot(hist)
# # plt.show()

# img_eq=cv2.equalizeHist(img)
# cv2.imshow("", img_eq)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# hist_e=cv2.calcHist([img_eq],[0],None,[256],[0,256])
# plt.plot(hist_e)
# plt.show()

#==============γ変換============
##画像の明るさの変換方法
##https://www.stjun.com/entry/2020/02/08/183341

# gamma=0.4
# img = cv2.imread("data/src/Lena.jpg")

# gamma_cvt= np.zeros((256,1),dtype=np.uint8)
# for i in range(256):
#     gamma_cvt[i][0]=256*(float(i)/255)**(1.0/gamma)

# img_gamma= cv2.LUT(img,gamma_cvt)

# cv2.imshow("", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("", img_gamma)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


#==============トラックバー============
# def onTrackbar(position):
#     global trackValue
#     trackValue= position 

# trackValue=100
# cv2.namedWindow("img")
# cv2.createTrackbar("track", "img", trackValue,255,onTrackbar)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#==============図形の描画・文字の記述============
# img=np.ones((500,500,3))*500

# cv2.line(img, (0,0), (150,190), (255,0,0),5)
# cv2.rectangle(img, (100,25), (300,150), (0,255,0),5)
# cv2.circle(img, (100,100),55, (0,0,255),-1)
# font=cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,"OpenCV", (100,300),font,1,(0,255,0),3,cv2.LINE_AA)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#==============二値化============

# img = cv2.imread("data/src/grapes.jpg",0)
# cv2.imshow("", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# threshold = 100
# ret ,img_th=cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
# cv2.imshow("", img_th)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ret2 ,img_o=cv2.threshold(img,0,255,cv2.THRESH_OTSU)

# # hist=cv2.calcHist([img],[0],None,[256],[0,256])
# # plt.plot(hist)
# # plt.show()

# img_ada=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,1)
# cv2.imshow("otsu", img_o)
# cv2.imshow("ada", img_ada)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#==============二値化+トラックバー============

# img = cv2.imread("data/src/cone2.jpg",0)
# cv2.imshow("", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#トラックバー
# def onTrackbar(position):
#     global threshold
#     threshold= position 


# cv2.namedWindow("img")
# threshold=100
# cv2.createTrackbar("track", "img", threshold,255,onTrackbar)

# while True:
#     ret ,img_th=cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
#     cv2.imshow("img", img_th)
#     cv2.imshow("scr", img)
#     if cv2.waitKey(10)==27:
#         break

# cv2.destroyAllWindows()

#==============アファイン変換============
#https://qiita.com/koshian2/items/c133e2e10c261b8646bf
# img = cv2.imread("data/src/grapes.jpg")
# h,w=img.shape[:2]

# dx,dy=30,30

# # afn_mat= np.float32([[1,0,dx],[0,1,dy]])
# # img_afn=cv2.warpAffine(img,afn_mat,(w,h))

# # cv2.imshow("trans", img_afn)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# rot_mat=cv2.getRotationMatrix2D((w/2,h/2),180,1)

# img_afn2=cv2.warpAffine(img,rot_mat,(w,h))
# cv2.imshow("trans", img_afn2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#==============透視変換============
#https://di-acc2.com/programming/python/19094/

# img = cv2.imread("data/src/drive.jpg")
# h,w=img.shape[:2]
# cv2.imshow("", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# par1=np.float32([[100,500],[300,500],[300,100],[100,100]])
# par2=np.float32([[100,500],[300,500],[280,200],[150,200]])

# psp_matrix=cv2.getPerspectiveTransform(par1,par2)
# img_psp= cv2.warpPerspective(img,psp_matrix,(w,h))

# cv2.imshow("", img_psp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#==============畳み込み============
#https://cvml-expertguide.net/terms/cv/image-filtering/convolution-for-image-filtering/

# kernel=np.ones((3,3))/9.0 

# img = cv2.imread("data/src/Lena.jpg",0)
# # img_ke1 = cv2.filter2D(img,-1,kernel)
# # cv2.imshow("img",img_ke1)
# # cv2.imshow("src",img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# kernel2=np.zeros((3,3))
# kernel2[0,0]=1
# kernel2[1,0]=2
# kernel2[2,0]=1
# kernel2[0,2]=-1
# kernel2[1,2]=-2
# kernel2[2,2]=-1


# img_ke2=cv2.filter2D(img,-1,kernel2)
# cv2.imshow("img",img_ke2)
# cv2.imshow("src",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#==============平滑化============
# img = cv2.imread("data/src/buildings.jpg")
# img_blur= cv2.blur(img,(3,3))

# cv2.imshow("img",img_blur)
# cv2.imshow("src",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#ガウシアンフィルタ
#https://www.nomuyu.com/gaussian-filter/#st-toc-h-1
# img_ga=cv2.GaussianBlur(img,(9,9),2)

# cv2.imshow("img",img_ga)
# cv2.imshow("src",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#メディアンフィルタ
#https://algorithm.joho.info/image-processing/median-filter/
# img_me=cv2.medianBlur(img,5)

# cv2.imshow("img", img_me)
# cv2.imshow("src",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#バイラテラルフィルタ
#https://deecode.net/?p=526
#エッジを残したまま平滑化をしたい

# img_bi=cv2.bilateralFilter(img,20,30,30)

# cv2.imshow("img", img_bi)
# cv2.imshow("src",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#==============エッジの検出============
#http://www.thothchildren.com/chapter/5b4b6d05103f2f316870f762

#ソーベルフィルタ
# img = cv2.imread("data/src/Lena.jpg")
# # cv2.imshow("src",img)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # img_sobelx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)
# # img_sobely=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)

# # img_sobelx=cv2.convertScaleAbs(img_sobelx)
# # img_sobely=cv2.convertScaleAbs(img_sobely)



# # cv2.imshow("x", img_sobelx)
# # cv2.imshow("y",img_sobely)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# #ラプラシアンフィルタ
# img_lap=cv2.Laplacian(img,cv2.CV_32F)
# img_lap=cv2.convertScaleAbs(img_lap)
# img_lap*=2

# img_blur= cv2.GaussianBlur(img,(3,3),2)
# img_lap2=cv2.Laplacian(img_blur,cv2.CV_32F)
# img_lap2=cv2.convertScaleAbs(img_lap2)
# img_lap2*=4
# cv2.imshow("lap*2",img_lap)
# cv2.imshow("Blur",img_lap2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#==============エッジの検出(Canny)============
#==流れ==
# 1.ガウシアンフィルタでぼかす(ノイズを取り除く)
# 2.Sobelフィルターで微分する(x,y方向)
# 3.極大点を探す
# 4.二段階の閾値処理でエッジを残す

# img = cv2.imread("data/src/cone1.jpg")

# img_canny=cv2.Canny(img,10,180)
# img_canny2=cv2.Canny(img,100,200)
# cv2.imshow("Canny",img_canny)
# cv2.imshow("Canny2",img_canny2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#==============直線と円の検出============

#直線の検出
# img = cv2.imread("data/src/cone1.jpg")
# img_g = cv2.imread("data/src/cone2.jpg",0)

# # cv2.imshow("img",img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # lines= cv2.HoughLines(img,1,np.pi/180,350)

# img_canny=cv2.Canny(img,360,550)

# cv2.imshow("img",img_canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# lines= cv2.HoughLines(img_canny,1,np.pi/180,100)

# for i in lines[:]:
#     rho=i[0][0]
#     theta=i[0][1]
#     a=np.cos(theta)
#     b=np.sin(theta)
#     x0=rho*a
#     y0=rho*b

#     x1=int(x0+1000*(-b))
#     y1=int(y0+1000*(a))
#     x2=int(x0-1000*(-b))
#     y2=int(y0-1000*(a))

#     cv2.line(img, (x1,y1), (x2,y2), (255,0,0),1)


# cv2.imshow("img",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#円の検出

# img2 = cv2.imread("data/src/grapes.jpg")
# img2_g = cv2.imread("data/src/grapes.jpg",0)

# circles=cv2.HoughCircles(img2_g,cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=20,param2=35,minRadius=1,maxRadius=30)

# for i in circles[0]:
#     cv2.circle(img2,(i[0],i[1]),i[2],(255,0,0),1)

# cv2.imshow("img",img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#==============特徴抽出============
#https://qiita.com/icoxfog417/items/adbbf445d357c924b8fc
# img = cv2.imread("data/src/buildings.jpg")
# img_g=cv2.imread("data/src/buildings.jpg",0)

# img_harris=copy.deepcopy(img)
# img_dst=cv2.cornerHarris(img_g,2,3,0.04)

# img_harris[img_dst > 0.05*img_dst.max()]=[0,0,255]

# cv2.imshow("img",img_harris)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img_kaze=copy.deepcopy(img)
# kaze=cv2.KAZE_create()

# kp1=kaze.detect(img,None)
# img_kaze=cv2.drawKeypoints(img_kaze,kp1,None)

# cv2.imshow("img",img_kaze)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img_kaze=copy.deepcopy(img)
# kaze=cv2.AKAZE_create()

# kp1=kaze.detect(img,None)
# img_kaze=cv2.drawKeypoints(img_kaze,kp1,None)

# cv2.imshow("img",img_kaze)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #orb
# img_orb=copy.deepcopy(img)
# orb=cv2.ORB_create()

# kp2=orb.detect(img,None)
# img_orb=cv2.drawKeypoints(img_orb,kp2,None)

# cv2.imshow("img",img_orb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#==============ブロブ検出============

# img = cv2.imread("data/src/Blob.png")
# img_g=cv2.imread("data/src/Blob.png",0)

# # cv2.imshow("img",img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# #面積の重心に書く
# ret,img_bi=cv2.threshold(img_g,100,255,cv2.THRESH_BINARY)
# # cv2.imshow("img", img_bi)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# nLabels,labelImage,stats,centroids=cv2.connectedComponentsWithStats(img_bi)

# img_blob=copy.deepcopy(img)
# h,w=img_g.shape
# # print(h)
# # print(w)
# color=[[255,0,0],[0,255,0],[0,0,255],[255,255,0]]

# for y in range(h):
#     for x in range(w):
#         if labelImage[y,x]>0:
#             img_blob[y,x]=color[labelImage[y,x]-1]


# #centroids=重心
# for i in range(1,nLabels):
#     xc=int(centroids[i][0])
#     yc=int(centroids[i][1])
#     font=cv2.FONT_HERSHEY_COMPLEX
#     scale=1
#     color=(255,255,255)
#     cv2.putText(img_blob,str(stats[i][-1]),(xc,yc),font,scale,color )

# cv2.imshow("img", img_blob)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#==============輪郭の検出============
# img = cv2.imread("data/src/cone1.jpg")
# img_g=cv2.imread("data/src/cone1.jpg",0)

# img = cv2.imread("data/src/Blob.png")
# img_g=cv2.imread("data/src/Blob.png",0)
# ret,img_bi=cv2.threshold(img_g,100,255,cv2.THRESH_BINARY)

# img_con,contours,hierarchy=cv2.findContours(img_bi,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# img_contour=cv2.drawContours(img,contours,-1,(255,0,0),1)

# cv2.imshow("img", img_contour)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#==============色検出============
cap = cv2.VideoCapture("data/movie/Mobility.mp4")

while True:
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", 640,480)
    ret, frame=cap.read()
    if ret==False:
        break
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower=np.array([20,50,50])
    upper=np.array([25,255,255])
    frame_mask=cv2.inRange(hsv,lower,upper)
    dst= cv2.bitwise_and(frame,frame,mask=frame_mask)

    cv2.imshow("img",dst)
    if cv2.waitKey(10)==27:
        break

cv2.destroyAllWindows()

    