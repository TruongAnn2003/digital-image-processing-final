import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
L=256

#Chuyển đổi sang ảnh âm
def Negative(imgin):
    if len(imgin.shape) == 2:
        return Negative_GrayScale(imgin)
    else:
        return Negative_Color(imgin)

def Negative_GrayScale(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N),np.uint8)+255
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            s= L - 1 -r
            imgout[x,y]=np.uint8(s) 
    return imgout

def Negative_Color(imgin):
    M,N,C = imgin.shape
    imgout = np.zeros((M,N,C),np.uint8)+255
    for x in range(0,M):
        for y in range(0,N):
            b=imgin[x,y,0]
            g=imgin[x,y,1]
            r=imgin[x,y,2]
            
            b = L - 1 - b
            g = L - 1 - g
            r = L - 1 - r
            
            imgout[x,y,0]=np.uint8(b) 
            imgout[x,y,1]=np.uint8(g) 
            imgout[x,y,2]=np.uint8(r) 
    return imgout

'''
- Tăng vùng tối, nén vùng sáng
- Tăng độ tương phản
'''
def Logarit(imgin, c=None):
    if len(imgin.shape) == 2:
        return Logarit_GrayScale(imgin, c)
    else:
        return Logarit_Color(imgin, c)

def Logarit_GrayScale(imgin, c=None):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    if c is None:
        c = (L - 1) / np.log(1.0 * L)
    for x in range(M):
        for y in range(N):
            r = imgin[x, y]
            s = c * np.log(1.0 + r)
            imgout[x, y] = np.uint8(s)
    return imgout

def Logarit_Color(imgin, c=None):
    if c is None:
        c = (L - 1) / np.log(1.0 * L)
    channels = cv2.split(imgin)
    imgout_channels = [Logarit_GrayScale(channel, c) for channel in channels]
    imgout = cv2.merge(imgout_channels)
    return imgout

'''
- Chỉnh độ tương phản, độ sáng
'''
def Power(imgin,gamma=None):
    if len(imgin.shape) == 2:
        return Power_GrayScale(imgin,gamma)
    else:
        return Power_Color(imgin,gamma)

def Power_GrayScale(imgin,gamma=None):
    M,N = imgin.shape
    imgout = np.zeros((M,N),np.uint8)+255
    if gamma is None:
        gamma = 5.0
    c = np.power(L-1,1-gamma)
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            s= c * np.power(r,gamma)
            imgout[x,y]=np.uint8(s) 
    return imgout

def Power_Color(imgin,gamma=None):
    if gamma is None:
        gamma = 5.0
    channels = cv2.split(imgin)
    imgout_channels = [Power_GrayScale(channel,gamma) for channel in channels]
    imgout = cv2.merge(imgout_channels)
    return imgout

def PiecewiseLinearGray(imgin):
    if len(imgin.shape) == 2:
        return HandlePiecewiseLinearGray(imgin)
    else:
        img_gray= cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return HandlePiecewiseLinearGray(img_gray)

def HandlePiecewiseLinearGray(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N),np.uint8)+255
    rmin, rmax,_,_= cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            #Đoạn I
            if r < r1:
                s = s1*r/r1
            #Đoạn II
            elif r < r2:
                s = (s2-s1)*(r-r1)/(r2-r1)+s1
            #Đoạn III
            else:
                s = (L-1-s2)*(r-r2)/(L-1-r2)+s2
            imgout[x,y]=np.uint8(s) 
    return imgout

def Histogram(imgin):
    if len(imgin.shape) == 2:
        return Histogram_GrayScale(imgin)
    else:
        return Histogram_Color(imgin)

def Histogram_GrayScale(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,L),np.uint8)+255
    h = np.zeros(L,np.int32)
    for x in range(0,M):
        for y in range(0,N):
            r = imgin[x,y]
            h[r] = h[r]+1
    p = h /(M*N)
    scale = 3000 
    for r in range(0,L):
        cv2.line(imgout,(r,M-1),(r,M-1-int(scale*p[r])),(0,0,0))
    return imgout

def Histogram_Color(imgin):
    img_hsv = cv2.cvtColor(imgin, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # Tính toán histogram cho từng kênh màu
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])

    # Vẽ histogram
    plt.figure()
    plt.plot(hist_h, color='r', label='Hue(màu sắc)')
    plt.plot(hist_s, color='g', label='Saturation(độ bão hòa)')
    plt.plot(hist_v, color='b', label='Value(độ sáng)')
    plt.xlim([0, 256])
    plt.xlabel('Bins')
    plt.ylabel('Pixel Count')
    plt.title('Histogram')
    plt.legend()
    plt.savefig('histogram.png')
    # Đọc biểu đồ đã lưu và trả về
    histogram_img = cv2.imread('histogram.png')
    return histogram_img

def HistEqual(imgin):
    if len(imgin.shape) == 2:
        return HistEqual_GrayScale(imgin)
    else:
        return HistEqual_Color(imgin)

def HistEqual_GrayScale(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N),np.uint8)
    h = np.zeros(L,np.int32)
    print(h.size)
    for x in range(0,M):
        for y in range(0,N):
            r = imgin[x,y]
            h[r] = h[r]+1
    p = h /(M*N)
    s = np.zeros(L,np.float32)
    for k in range(0,L):
        for j in range(0,k+1):
            s[k] = s[k]+p[j]
        s[k] =  (L-1)*s[k]
    
    for x in range(0,M):
        for y in range(0,N):
            r = imgin[x,y]
            imgout[x,y] = np.uint8(s[r])      
    return imgout

def HistEqual_Color(imgin):
    hsv = cv2.cvtColor(imgin, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Cân bằng histogram cho các kênh màu Hue và Saturation
    h_eq = cv2.equalizeHist(h)
    s_eq = cv2.equalizeHist(s)
    
    # Kết hợp các kênh đã cân bằng lại để tạo ra ảnh màu đã cân bằng histogram
    hsv_eq = cv2.merge([h_eq, s_eq, v])
    
    # Chuyển đổi ảnh trở lại không gian màu BGR
    img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    
    return img_eq

def EnhanceColor(image, red_factor, green_factor, blue_factor):
    enhanced_image = image.copy()
    if len(image.shape) == 2:
        enhanced_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
    enhanced_image[:, :, 0] = np.clip(enhanced_image[:, :, 0] * blue_factor, 0, 255)
    enhanced_image[:, :, 1] = np.clip(enhanced_image[:, :, 1] * green_factor, 0, 255)
    enhanced_image[:, :, 2] = np.clip(enhanced_image[:, :, 2] * red_factor, 0, 255)
    return enhanced_image