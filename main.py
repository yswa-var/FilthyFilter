import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path


class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")

        # Initialize variables
        self.current_image = None
        self.original_image = None
        self.current_image_path = None

        self.setup_ui()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.load_btn = ttk.Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=0, column=0, padx=5)

        self.save_btn = ttk.Button(self.button_frame, text="Save Image", command=self.save_image)
        self.save_btn.grid(row=0, column=1, padx=5)

        self.reset_btn = ttk.Button(self.button_frame, text="Reset", command=self.reset_image)
        self.reset_btn.grid(row=0, column=2, padx=5)

        self.filters_frame = ttk.LabelFrame(self.main_frame, text="Filters", padding="5")
        self.filters_frame.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        filters = [
            ("Grayscale", self.apply_grayscale),
            ("Blur", self.apply_blur),
            ("Sharpen", self.apply_sharpen),
            ("Edge Detection", self.apply_edge_detection),
            ("Sepia", self.apply_sepia)
        ]

        for i, (filter_name, filter_command) in enumerate(filters):
            btn = ttk.Button(self.filters_frame, text=filter_name, command=filter_command)
            btn.grid(row=0, column=i, padx=5)

        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=2, column=0, padx=5, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            self.current_image_path = file_path
            self.original_image = Image.open(file_path)
            self.current_image = self.original_image.copy()
            self.display_image()

    def save_image(self):
        if self.current_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
            )
            if file_path:
                self.current_image.save(file_path)

    def reset_image(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.display_image()

    def display_image(self):
        if self.current_image:
            display_size = (800, 600)
            self.current_image.thumbnail(display_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(self.current_image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference

    def apply_grayscale(self):
        if self.current_image:
            img_array = np.array(self.current_image)
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            self.current_image = Image.fromarray(gray_image)
            self.display_image()

    def apply_blur(self):
        if self.current_image:
            img_array = np.array(self.current_image)
            blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
            self.current_image = Image.fromarray(blurred)
            self.display_image()

    def apply_sharpen(self):
        if self.current_image:
            sharpening_kernel = np.array([[-1, -1, -1],
                                          [-1, 9, -1],
                                          [-1, -1, -1]])
            img_array = np.array(self.current_image)
            sharpened = cv2.filter2D(img_array, -1, sharpening_kernel)
            self.current_image = Image.fromarray(sharpened)
            self.display_image()

    def apply_edge_detection(self):
        if self.current_image:
            img_array = np.array(self.current_image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            self.current_image = Image.fromarray(edges_rgb)
            self.display_image()

    def apply_sepia(self):
        if self.current_image:
            img_array = np.array(self.current_image)
            sepia_kernel = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            sepia_image = cv2.transform(img_array, sepia_kernel)
            sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
            self.current_image = Image.fromarray(sepia_image)
            self.display_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()