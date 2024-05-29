import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

class ImageFilterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Filter App")

        self.original_image = None
        self.filtered_image = None

        self.create_widgets()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.master, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image display frame
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.grid(row=0, column=0, padx=10, pady=10)

        # Original image label
        self.original_label = ttk.Label(self.image_frame, text="Ảnh Gốc")
        self.original_label.grid(row=0, column=0)

        self.original_canvas = tk.Canvas(self.image_frame, width=400, height=400)
        self.original_canvas.grid(row=1, column=0)

        # Filtered image label
        self.filtered_label = ttk.Label(self.image_frame, text="Ảnh sau khi lọc")
        self.filtered_label.grid(row=0, column=1)

        self.filtered_canvas = tk.Canvas(self.image_frame, width=400, height=400)
        self.filtered_canvas.grid(row=1, column=1)

        # Control frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=1, column=0, padx=10, pady=10)

        # Buttons
        self.open_button = ttk.Button(self.control_frame, text="Chọn Ảnh", command=self.open_image)
        self.open_button.grid(row=0, column=0, padx=5, pady=5)

        self.apply_button = ttk.Button(self.control_frame, text="Áp Dụng Lọc", command=self.apply_filter)
        self.apply_button.grid(row=0, column=1, padx=5, pady=5)

        self.save_button = ttk.Button(self.control_frame, text="Lưu Ảnh", command=self.save_image)
        self.save_button.grid(row=0, column=2, padx=5, pady=5)

        # Combobox with default value
        self.filter_combobox = ttk.Combobox(self.control_frame, state="readonly", width=20)
        self.filter_combobox.grid(row=0, column=3, padx=5, pady=5)
        self.filter_combobox['values'] = [
            "Smooth/Low Pass Filter",
            "Sharpening/High Pass Filter",
            "Laplacian Filter",
            "Ideal Low Pass Filter",
            "Gaussian Low Pass Filter",
            "Butterworth Lowpass Filter",
            "Ideal High Pass Filter",
            "Gaussian High Pass Filter",
            "Butterworth Highpass Filter"
        ]
        self.filter_combobox.set("Smooth/Low Pass Filter")  # Set default value

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        if file_path:
            self.original_image = cv2.imread(file_path, 0)
            self.display_image(self.original_image, canvas=self.original_canvas)

    def display_image(self, img, canvas):
        # Lấy kích thước của hình ảnh gốc
        height, width = img.shape[:2]

        # Thiết lập kích thước của canvas
        canvas.config(width=width, height=height)

        # Chuyển đổi định dạng màu và tạo đối tượng Image từ mảng numpy
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Tạo đối tượng ImageTk để hiển thị trên canvas
        img_tk = ImageTk.PhotoImage(image=img)

        # Gán đối tượng ImageTk vào thuộc tính của canvas
        canvas.img_tk = img_tk

        # Xóa hình ảnh hiện tại trên canvas (nếu có)
        canvas.delete("all")

        # Vẽ hình ảnh mới trên canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    
    # Functions for different filters
    def laplacian(self, f):
        new_img = cv2.Laplacian(f, -1)  # Apply Laplacian filter
        new_img2 = cv2.subtract(f, new_img)  # Subtract original image from Laplacian filter result
        return new_img2

    def ideal_lowpass(self, f):
        rows, cols = f.shape
        crow, ccol = rows // 2, cols // 2

        # Create Ideal Low Pass Filter
        ideal_filter = np.zeros((rows, cols), np.uint8)
        radius = 50  # Cutoff frequency radius
        ideal_filter[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1

        # Apply spatial domain and compute FFT
        f_shift = np.fft.fftshift(np.fft.fft2(f))
        f_filtered = f_shift * ideal_filter
        f_filtered_shift = np.fft.ifftshift(f_filtered)

        # Perform inverse FFT to get filtered image
        filtered_image = np.abs(np.fft.ifft2(f_filtered_shift))
        filtered_image = np.uint8(filtered_image)
        return filtered_image

        # Hàm lọc thông thấp Ideal
    def ideal_lowpass(self, f):
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
    def gaussian_lowpass(self, f):
        # Thực hiện lọc thông thấp Gauss
        filtered_image = cv2.GaussianBlur(f, (5, 5), 0)
        return filtered_image

    # Hàm lọc thông thấp Butterworth
    def butterworth_lowpass(self, f):
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
    def ideal_highpass(self, f):
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
    def gaussian_highpass(self, f):
        # Thực hiện lọc thông cao Gauss
        lowpass_filtered = cv2.GaussianBlur(f, (5, 5), 0)
        filtered_image = cv2.subtract(f, lowpass_filtered)
        return filtered_image

    # Hàm lọc thông cao Butterworth
    def butterworth_highpass(self, f):
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


    def apply_filter(self):
        selected_filter = self.filter_combobox.get()

        if self.original_image is None:
            messagebox.showerror("Error", "Please open an image first!")
            return

        if selected_filter == "":
            messagebox.showerror("Error", "Please select a filter!")
            return

        if selected_filter == "Smooth/Low Pass Filter":
            kernel = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]) / 9
        elif selected_filter == "Sharpening/High Pass Filter":
            kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
        elif selected_filter == "Laplacian Filter":
            self.filtered_image = self.laplacian(self.original_image)
        elif selected_filter == "Ideal Low Pass Filter":
            filtered_image = self.ideal_lowpass(self.original_image)
            if filtered_image is not None:  # Sửa tại đây
                self.filtered_image = filtered_image  # Sửa tại đây
                self.display_image(self.filtered_image, canvas=self.filtered_canvas)  # Hiển thị kết quả
            return
        elif selected_filter == "Gaussian Low Pass Filter":
            filtered_image = self.gaussian_lowpass(self.original_image)
            if filtered_image is not None:  # Sửa tại đây
                self.filtered_image = filtered_image  # Sửa tại đây
                self.display_image(self.filtered_image, canvas=self.filtered_canvas)  # Hiển thị kết quả
            return
        elif selected_filter == "Butterworth Lowpass Filter":
            filtered_image = self.butterworth_lowpass(self.original_image)
            if filtered_image is not None:  # Sửa tại đây
                self.filtered_image = filtered_image  # Sửa tại đây
                self.display_image(self.filtered_image, canvas=self.filtered_canvas)  # Hiển thị kết quả
            return
        elif selected_filter == "Ideal High Pass Filter":
            filtered_image = self.ideal_highpass(self.original_image)
            if filtered_image is not None:  # Sửa tại đây
                self.filtered_image = filtered_image  # Sửa tại đây
                self.display_image(self.filtered_image, canvas=self.filtered_canvas)  # Hiển thị kết quả
            return
        elif selected_filter == "Gaussian High Pass Filter":
            filtered_image = self.gaussian_highpass(self.original_image)
            if filtered_image is not None:  # Sửa tại đây
                self.filtered_image = filtered_image  # Sửa tại đây
                self.display_image(self.filtered_image, canvas=self.filtered_canvas)  # Hiển thị kết quả
            return
        elif selected_filter == "Butterworth Highpass Filter":
            filtered_image = self.butterworth_highpass(self.original_image)
            if filtered_image is not None:  # Sửa tại đây
                self.filtered_image = filtered_image  # Sửa tại đây
                self.display_image(self.filtered_image, canvas=self.filtered_canvas)  # Hiển thị kết quả
            return
        else:
            return

        # Apply filter to original image
        if selected_filter != "Laplacian Filter" and selected_filter != "Result after Laplacian":
            self.filtered_image = cv2.filter2D(self.original_image, -1, kernel)

        # Display filtered image
        self.display_image(self.filtered_image, canvas=self.filtered_canvas)


    def save_image(self):
        if self.filtered_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
            if file_path:
                cv2.imwrite(file_path, self.filtered_image)
                messagebox.showinfo("Save Image", "Image saved successfully!")
        else:
            messagebox.showwarning("Save Image", "No filtered image to save!")

def main():
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
