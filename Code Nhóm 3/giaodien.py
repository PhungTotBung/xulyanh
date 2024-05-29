import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from ttkthemes import ThemedStyle
import subprocess  

# Tạo cửa sổ
window = tk.Tk()
window.title("Nhóm 3")

# Đặt kích thước cửa sổ và giữ trung tâm màn hình
window_width = 400
window_height = 350
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Thay đổi giao diện nút sử dụng ThemedStyle
style = ThemedStyle(window)
style.set_theme('clearlooks')  # Thay đổi theme nếu cần
    
# Tiêu đề
title_label = ttk.Label(window, text="Chọn chức năng bạn muốn thực hiện", font=("Arial", 16))
title_label.pack(pady=10)

# Hình ảnh
image_path = "bg.png"  # Thay đổi đường dẫn tới hình ảnh của bạn
if os.path.exists(image_path):
    image = Image.open(image_path)
    image.thumbnail((150, 150))
    photo = ImageTk.PhotoImage(image)

    image_label = ttk.Label(window, image=photo)
    image_label.image = photo
    image_label.pack()

# Hàm ẩn cửa sổ chính
def hide_window():
    window.withdraw()

# Hàm hiển thị cửa sổ chính
def show_window():
    window.deiconify()

def chonanh():
    hide_window()
    process = subprocess.Popen(["python", "chonanh.py"])
    process.wait()
    show_window()

def chonfolder():
    hide_window()
    process = subprocess.Popen(["python", "chonfolder.py"])
    process.wait()
    show_window()

def sift():
    hide_window()
    process = subprocess.Popen(["python", "SIFT.py"])
    process.wait()
    show_window()

def doi_size():
    hide_window()
    process = subprocess.Popen(["python", "doi_size.py"])
    process.wait()
    show_window()

# Nút nâng cao chất lượng 1 hình ảnh
load_folder_button = ttk.Button(window, text="Nâng cao 1 ảnh", command=chonanh, width=20)
load_folder_button.pack(pady=10)

# Nút nâng cao chất lượng 1 folder
zone_button = ttk.Button(window, text="Nâng cao 1 folder", command=chonfolder, width=20)
zone_button.pack(pady=10)

# Nút rút trích đặc trưng ảnh 
image_button = ttk.Button(window, text="SIFT", command=sift, width=20)
image_button.pack(pady=10)

# Nút chuẩn hóa kích thước tất cả ảnh của 1 folder
image_button = ttk.Button(window, text="Chuẩn hóa kích thước ảnh", command=doi_size, width=20)
image_button.pack(pady=10)

# Hiển thị cửa sổ
window.mainloop()
