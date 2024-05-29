import os
import cv2
import PIL
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Khai báo biến toàn cục
images = []  # Danh sách các hình ảnh trong thư mục
current_image_index = 0  # Chỉ mục của hình ảnh hiện tại

# Hàm mở thư mục chứa các hình ảnh
def open_folder():
    global folder_path
    global images
    global current_image_index
    
    folder_path = filedialog.askdirectory()  # Mở hộp thoại để chọn thư mục
    if folder_path:
        # Lấy danh sách các hình ảnh trong thư mục và lưu vào biến images
        images = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
                  if filename.endswith(('.jpg', '.png', '.jpeg'))]
        
        if images:
            current_image_index = 0  # Đặt chỉ mục của hình ảnh hiện tại về 0
            display_image()  # Hiển thị hình ảnh đầu tiên
        else:
            messagebox.showwarning("Cảnh báo", "Thư mục không chứa hình ảnh.")

# Hàm hiển thị hình ảnh tại chỉ mục hiện tại
def display_image():
    global images
    global current_image_index
    
    if images:
        image_path = images[current_image_index]
        image = Image.open(image_path)
        image = image.resize((image.width, image.height), PIL.Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

# Hàm để chuyển đổi sang hình ảnh tiếp theo trong danh sách
def next_image():
    global current_image_index
    if images:
        current_image_index = (current_image_index + 1) % len(images)  # Lặp lại nếu đã đến cuối danh sách
        display_image()  # Hiển thị hình ảnh mới

# Hàm áp dụng bộ lọc cho tất cả các hình ảnh trong thư mục đã chọn
def apply_filter_to_folder():
    global images
    global folder_path
    
    selected_filter = filter_combobox.get()  # Lấy bộ lọc được chọn từ combobox
    if not os.path.isdir(folder_path):
        return
    
    output_folder_path = os.path.join(folder_path, selected_filter.replace(" ", "_") + "_filtered_images")
    os.makedirs(output_folder_path, exist_ok=True)  # Tạo thư mục mới để lưu hình ảnh đã lọc
    
    for image_path in images:
        # Đọc hình ảnh
        original_image = cv2.imread(image_path, 0)

        # Áp dụng bộ lọc tương ứng
        if selected_filter == "Smooth/Low Pass Filter":
            filtered_image = smooth_lowpass(original_image)
        elif selected_filter == "Sharpening/High Pass Filter":
            filtered_image = sharpening_highpass(original_image)
        elif selected_filter == "Laplacian Filter":
            filtered_image = laplacian(original_image)
        elif selected_filter == "Result after Laplacian":
            filtered_image = result_after_laplacian(original_image)
        elif selected_filter == "Ideal Low Pass Filter":
            filtered_image = ideal_lowpass(original_image)
        elif selected_filter == "Gaussian Low Pass Filter":
            filtered_image = gaussian_lowpass(original_image)
        elif selected_filter == "Butterworth Lowpass Filter":
            filtered_image = butterworth_lowpass(original_image)
        elif selected_filter == "Ideal High Pass Filter":
            filtered_image = ideal_highpass(original_image)
        elif selected_filter == "Gaussian High Pass Filter":
            filtered_image = gaussian_highpass(original_image)
        elif selected_filter == "Butterworth Highpass Filter":
            filtered_image = butterworth_highpass(original_image)
        else:
            filtered_image = None
        
        if filtered_image is not None:
            # Lưu hình ảnh đã được lọc vào thư mục mới
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_folder_path, "filtered_" + filename)
            cv2.imwrite(output_path, filtered_image)
    
    messagebox.showinfo("Hoàn thành", "Đã áp dụng bộ lọc cho tất cả các hình ảnh trong thư mục!")

# Hàm thực hiện bộ lọc Laplacian
def result_after_laplacian(f):
    new_img = cv2.Laplacian(f, -1)  # Áp dụng bộ lọc Laplacian
    new_img2 = cv2.subtract(f, new_img)  # Trừ hình ảnh gốc với kết quả của bộ lọc Laplacian

    return new_img2

# Hàm lọc thông thấp Gauss
def smooth_lowpass(f):
    # Thực hiện lọc thông thấp Gauss
    filtered_image = cv2.GaussianBlur(f, (5, 5), 0)
    return filtered_image

# Hàm lọc thông cao để làm sắc nét hình ảnh
def sharpening_highpass(f):
    # Định nghĩa kernel cho bộ lọc làm sắc nét
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    # Áp dụng bộ lọc
    sharpened_image = cv2.filter2D(f, -1, kernel)
    return sharpened_image

# Hàm thực hiện bộ lọc Laplacian
def laplacian(f):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
        # Áp dụng bộ lọc Laplacian
    filtered_image = cv2.filter2D(f, -1, kernel)
    laplacian = cv2.filter2D(f, -1, kernel)
    return laplacian


# Hàm lọc thông thấp Ideal
def ideal_lowpass(f):
    rows, cols = f.shape
    crow, ccol = rows // 2, cols // 2

    # Tạo một bộ lọc thông thấp Ideal
    ideal_filter = np.zeros((rows, cols), np.uint8)
    radius = 50  # Độ lớn của bán kính cắt tần số
    ideal_filter[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1

    # Áp dụng phổ biến không gian và tính FFT
    f_shift = np.fft.fftshift(np.fft.fft2(f))
    f_filtered = f_shift * ideal_filter
    f_filtered_shift = np.fft.ifftshift(f_filtered)

    # Thực hiện FFT ngược để lấy hình ảnh đã lọc
    filtered_image = np.abs(np.fft.ifft2(f_filtered_shift))
    filtered_image = np.uint8(filtered_image)
    return filtered_image

# Hàm lọc thông thấp Gauss
def gaussian_lowpass(f):
    # Thực hiện lọc thông thấp Gauss
    filtered_image = cv2.GaussianBlur(f, (5, 5), 0)
    return filtered_image

# Hàm lọc thông thấp Butterworth
def butterworth_lowpass(f):
    rows, cols = f.shape
    crow, ccol = rows // 2, cols // 2

    # Tạo một bộ lọc thông thấp Butterworth
    n = 2  # Bậc của bộ lọc
    d0 = 50  # Độ lớn của bán kính cắt tần số
    butterworth_filter = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            butterworth_filter[i, j] = 1 / (1 + ((np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)) / d0) ** (2 * n))
    # Áp dụng phổ biến không gian và tính FFT
    f_shift = np.fft.fftshift(np.fft.fft2(f))
    f_filtered = f_shift * butterworth_filter
    f_filtered_shift = np.fft.ifftshift(f_filtered)

    # Thực hiện FFT ngược để lấy hình ảnh đã lọc
    filtered_image = np.abs(np.fft.ifft2(f_filtered_shift))
    filtered_image = np.uint8(filtered_image)
    return filtered_image

# Hàm lọc thông cao Ideal
def ideal_highpass(f):
    rows, cols = f.shape
    crow, ccol = rows // 2, cols // 2

    # Tạo một bộ lọc thông cao Ideal
    ideal_filter = np.ones((rows, cols), np.uint8)
    radius = 50  # Độ lớn của bán kính cắt tần số
    ideal_filter[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0

    # Áp dụng phổ biến không gian và tính FFT
    f_shift = np.fft.fftshift(np.fft.fft2(f))
    f_filtered = f_shift * ideal_filter
    f_filtered_shift = np.fft.ifftshift(f_filtered)

    # Thực hiện FFT ngược để lấy hình ảnh đã lọc
    filtered_image = np.abs(np.fft.ifft2(f_filtered_shift))
    filtered_image = np.uint8(filtered_image)
    return filtered_image

# Hàm lọc thông cao Gauss
def gaussian_highpass(f):
    # Thực hiện lọc thông cao Gauss
    lowpass_filtered = cv2.GaussianBlur(f, (5, 5), 0)
    filtered_image = cv2.subtract(f, lowpass_filtered)
    return filtered_image

# Hàm lọc thông cao Butterworth
def butterworth_highpass(f):
    rows, cols = f.shape
    crow, ccol = rows // 2, cols // 2

    # Tạo một bộ lọc thông cao Butterworth
    n = 2  # Bậc của bộ lọc
    d0 = 50  # Độ lớn của bán kính cắt tần số
    butterworth_filter = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            butterworth_filter[i, j] = 1 / (1 + ((np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)) / d0) ** (2 * n))

    # Áp dụng phổ biến không gian và tính FFT
    f_shift = np.fft.fftshift(np.fft.fft2(f))
    f_filtered = f_shift * butterworth_filter
    f_filtered_shift = np.fft.ifftshift(f_filtered)

    # Thực hiện FFT ngược để lấy hình ảnh đã lọc
    filtered_image = np.abs(np.fft.ifft2(f_filtered_shift))
    filtered_image = np.uint8(filtered_image)
    return filtered_image


# Tạo giao diện người dùng
root = tk.Tk()
root.title("Nhóm 3 - lọc nhiều ảnh")

main_frame = ttk.Frame(root, padding="20")
main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Label để hiển thị hình ảnh
image_label = ttk.Label(main_frame)
image_label.grid(column=0, row=0, columnspan=2)

# Button để mở thư mục và hiển thị hình ảnh đầu tiên
open_button = ttk.Button(main_frame, text="Chọn Thư Mục", command=open_folder)
open_button.grid(column=0, row=1, pady=10)

# Button để chuyển đổi sang hình ảnh tiếp theo
next_button = ttk.Button(main_frame, text="Hình Ảnh Tiếp Theo", command=next_image)
next_button.grid(column=1, row=1, pady=10)

filter_label = ttk.Label(main_frame, text="Chọn Bộ Lọc:")
filter_label.grid(column=0, row=2)

filters = ["Smooth/Low Pass Filter", "Sharpening/High Pass Filter", "Laplacian Filter", "Result after Laplacian",
           "Ideal Low Pass Filter", "Gaussian Low Pass Filter", "Butterworth Lowpass Filter",
           "Ideal High Pass Filter", "Gaussian High Pass Filter", "Butterworth Highpass Filter"]
filter_combobox = ttk.Combobox(main_frame, values=filters, state="readonly")
filter_combobox.grid(column=1, row=2)

apply_button = ttk.Button(main_frame, text="Apply Filter", command=apply_filter_to_folder)
apply_button.grid(column=0, row=3, columnspan=2, pady=10)

root.mainloop()
