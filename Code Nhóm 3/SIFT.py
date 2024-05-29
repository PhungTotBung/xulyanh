import cv2
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog

def select_image():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    file_path = filedialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
    return file_path

def Thuat_toan_SIFT(img1,img2):
    sift = cv2.SIFT_create()  #gọi phương thức để tìm các đặc trưng
    keypointt_1,descriptors_1 = sift.detectAndCompute(img1,None)    #dò tìm và mô tả đặc trưng ảnh 1
    keypointt_2,descriptors_2 = sift.detectAndCompute(img2,None)    #dò tìm và mô tả đặc trưng ảnh 2
    print("Tong so diem leypoint cua anh img1",len(keypointt_1))    #in tong so điểm đặc trưng ảnh 1
    print("Tong so diem keypoint cua anh img2",len(keypointt_2))

    keypointt_img1 = np.copy(img1)   #tạo ảnh chứa các điểm đặc trưng của ảnh 1
    keypointt_img2 = np.copy(img2)

    cv2.drawKeypoints(img1,keypointt_1,keypointt_img1,flags=2)   #vẽ các điểm đặc trưng của ảnh 1
    cv2.drawKeypoints(img2,keypointt_2,keypointt_img2,flags=2)

    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck = True)   #gọi phương thức BFMatcher để đối xứng các điểm đặc trưng

    matches = bf.match(descriptors_1,descriptors_2)   #thực hiện đối sánh các điểm đặc trưng
    matches = sorted(matches,key=lambda x:x.distance)    #sắp xếp điểm đối sánh
    print("Tong so diem keypoint doi sanh giua 2 anh: ",len(matches))

    img_result = cv2.drawMatches(img1,keypointt_1,img2,keypointt_2,matches,img2,flags=2)   #vẽ các điểm đối sánh
    return img_result,keypointt_img1,keypointt_img2   #trả về 3 ảnh chứa các đường đối sánh,các điểm đối sánh ảnh1 à ảnh 2

if __name__ == "__main__":
    print("Chọn 2 hình ảnh")
    img_path1 = select_image()
    img_path2 = select_image()

    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    img1RGB = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB) # chuyển ảnh sang hệ màu RGB
    img2RGB = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

    img_sokhop,img1_keypoint,img2_keypoint = Thuat_toan_SIFT(img1,img2) # gọi hàm so khớp

    fig = plt.figure(figsize=(16,9))   # tạo cửa sổ hiện thi ảnh gốc 1 và 2
    ax1,ax2 = fig.subplots(1,2)
    ax1.imshow(img1RGB)
    ax2.set_title("Anh goc 1")
    ax1.axis("off")

    ax2.imshow(img2RGB)
    ax2.set_title("Anh goc 2")
    ax2.axis("off")
    plt.show()   # hiển thị
    
    fig = plt.figure(figsize=(16,9))  # tạo cửa sổ hiện thi ảnh chứa điểm đặc trưng ảnh 1 và 2
    ax1,ax2 = fig.subplots(1,2)
    ax1.imshow(img1_keypoint)
    ax1.set_title("Anh 1 va keypioint")
    ax1.axis("off")

    ax2.imshow(img2_keypoint)
    ax2.set_title("Anh 2 va keypoints")
    ax2.axis("off")
    plt.show()

    fig = plt.figure(figsize=(16,9))     # tạo cửa sổ hiện thi ảnh so khớp của ảnh 1 và 2
    plt.imshow(img_sokhop)
    plt.title("Anh so khop")
    plt.axis("off")
    plt.show()