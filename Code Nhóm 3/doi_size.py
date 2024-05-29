from PIL import Image
import os
from tkinter import Tk, filedialog

def resize_images(source_folder, destination_folder, new_width):
    # Tạo thư mục đích nếu chưa tồn tại
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Lặp qua tất cả các tập tin trong thư mục nguồn
    for filename in os.listdir(source_folder):
        # Chỉ xử lý các tập tin hình ảnh
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Đường dẫn đầy đủ đến tập tin nguồn
            source_path = os.path.join(source_folder, filename)
            # Đọc hình ảnh
            img = Image.open(source_path)
            # Tính toán tỉ lệ kích thước mới
            width, height = img.size
            ratio = width / height
            new_height = int(new_width / ratio)
            # Thay đổi kích thước
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Lưu hình ảnh đã thay đổi kích thước vào thư mục đích
            destination_path = os.path.join(destination_folder, filename)
            img_resized.save(destination_path)
            print(f"Chuấn hóa kích thước ảnh {filename} thành côngs.")

def select_folder(prompt):
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    folder_path = filedialog.askdirectory(title=prompt) # show an "Open" dialog box and return the path to the selected folder
    return folder_path

if __name__ == "__main__":
    print("Chọn thư mục nguồn:")
    source_folder = select_folder("Chọn thư mục nguồn")
    print("Chọn thư mục đích:")
    destination_folder = select_folder("Chọn thư mục đích")
    new_width = 400

    # Gọi hàm để thay đổi kích thước của tất cả các tập tin trong thư mục nguồn và lưu vào thư mục đích
    resize_images(source_folder, destination_folder, new_width)
