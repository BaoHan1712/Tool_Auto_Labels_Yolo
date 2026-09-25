import os
import sys
import shutil
import random
import yaml
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def resource_path(relative_path):
    """Lấy đường dẫn tuyệt đối đến tài nguyên, hỗ trợ cả khi chạy dev và qua PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class YOLODatasetSplitterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("YOLO Dataset Splitter - Train / Valid / Test")
        self.geometry("1160x700")
        self.resizable(True, True)

        self.is_processing = False
        self._setup_ui()

    def _load_image(self, file_name, size):
        """Hàm load ảnh CTkImage an toàn hỗ trợ PyInstaller."""
        full_path = resource_path(file_name)
        if not os.path.exists(full_path):
            full_path = file_name

        if os.path.exists(full_path):
            try:
                pil_img = Image.open(full_path)
                return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=size)
            except Exception:
                pass
        return None

    def _setup_ui(self):
        # Header Tiêu đề
        title_label = ctk.CTkLabel(
            self, 
            text="HỆ THỐNG PHÂN CHIA TẬP DỮ LIỆU (SPLIT DATASET) - YOLO", 
            font=ctk.CTkFont(size=21, weight="bold")
        )
        title_label.pack(padx=20, pady=(15, 10))

        # Main Container chia 2 cột: Trái (Thao tác/Log), Phải (Tác giả)
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=15, pady=5)

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=0)
        container.grid_rowconfigure(0, weight=1)

        # ---------------- CỘT TRÁI (BẢNG ĐIỀU KHIỂN & CHIA DATA) ----------------
        left_frame = ctk.CTkFrame(container, fg_color="transparent")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        # Khung chọn thư mục nguồn & đích
        folder_frame = ctk.CTkFrame(left_frame)
        folder_frame.pack(fill="x", pady=(0, 8))
        folder_frame.grid_columnconfigure(1, weight=1)

        # 1. Thư mục done_labels nguồn
        ctk.CTkLabel(folder_frame, text="Thư mục 'done_labels':", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.entry_src = ctk.CTkEntry(folder_frame, placeholder_text="Chọn thư mục done_labels...")
        self.entry_src.grid(row=0, column=1, padx=5, pady=8, sticky="ew")
        ctk.CTkButton(folder_frame, text="Duyệt", width=75, command=self._browse_src).grid(row=0, column=2, padx=10, pady=8)

        # 2. Thư mục đích
        ctk.CTkLabel(folder_frame, text="Nơi lưu 'train_yolo_data':", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.entry_dest = ctk.CTkEntry(folder_frame, placeholder_text="Chọn thư mục chứa đầu ra...")
        self.entry_dest.grid(row=1, column=1, padx=5, pady=8, sticky="ew")
        ctk.CTkButton(folder_frame, text="Duyệt", width=75, command=self._browse_dest).grid(row=1, column=2, padx=10, pady=8)

        # Khung điều chỉnh tỷ lệ phần trăm
        ratio_frame = ctk.CTkFrame(left_frame)
        ratio_frame.pack(fill="x", pady=5)
        ratio_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ratio_frame, text="Tùy chỉnh tỷ lệ phân chia (%)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, columnspan=3, padx=10, pady=(8, 5), sticky="w")

        # Train ratio
        ctk.CTkLabel(ratio_frame, text="Train:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.slider_train = ctk.CTkSlider(ratio_frame, from_=0, to=100, number_of_steps=100, command=self._update_ratio_labels)
        self.slider_train.set(70)
        self.slider_train.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.lbl_train = ctk.CTkLabel(ratio_frame, text="70%", width=50)
        self.lbl_train.grid(row=1, column=2, padx=5, pady=5)

        # Valid ratio
        ctk.CTkLabel(ratio_frame, text="Valid:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.slider_val = ctk.CTkSlider(ratio_frame, from_=0, to=100, number_of_steps=100, command=self._update_ratio_labels)
        self.slider_val.set(20)
        self.slider_val.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.lbl_val = ctk.CTkLabel(ratio_frame, text="20%", width=50)
        self.lbl_val.grid(row=2, column=2, padx=5, pady=5)

        # Test ratio
        ctk.CTkLabel(ratio_frame, text="Test:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.slider_test = ctk.CTkSlider(ratio_frame, from_=0, to=100, number_of_steps=100, command=self._update_ratio_labels)
        self.slider_test.set(10)
        self.slider_test.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        self.lbl_test = ctk.CTkLabel(ratio_frame, text="10%", width=50)
        self.lbl_test.grid(row=3, column=2, padx=5, pady=5)

        # Tùy chọn xáo trộn & Tổng %
        opt_frame = ctk.CTkFrame(left_frame)
        opt_frame.pack(fill="x", pady=5)

        self.lbl_total_ratio = ctk.CTkLabel(opt_frame, text="Tổng: 100%", font=ctk.CTkFont(weight="bold"), text_color="#4CAF50")
        self.lbl_total_ratio.pack(side="left", padx=15, pady=8)

        self.chk_shuffle = ctk.CTkCheckBox(opt_frame, text="Xáo trộn ngẫu nhiên (Shuffle)")
        self.chk_shuffle.select()
        self.chk_shuffle.pack(side="right", padx=15, pady=8)

        # Progress bar & Status
        self.progress_bar = ctk.CTkProgressBar(left_frame)
        self.progress_bar.pack(fill="x", pady=(8, 3))
        self.progress_bar.set(0)

        self.lbl_status = ctk.CTkLabel(left_frame, text="Trạng thái: Sẵn sàng", text_color="gray70")
        self.lbl_status.pack(anchor="w", pady=(0, 4))

        # Console Logs
        self.txt_log = ctk.CTkTextbox(left_frame, height=130, font=("Consolas", 12))
        self.txt_log.pack(fill="both", expand=True, pady=4)

        # Button thực thi
        self.btn_run = ctk.CTkButton(
            left_frame, 
            text="Tạo thư mục & Chia tập dữ liệu", 
            font=ctk.CTkFont(size=15, weight="bold"),
            height=42,
            command=self._start_split_thread
        )
        self.btn_run.pack(fill="x", pady=(8, 5))

        # ---------------- CỘT PHẢI (THÔNG TIN TÁC GIẢ) ----------------
        right_frame = ctk.CTkFrame(container, width=280)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        right_frame.pack_propagate(False)

        author_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        author_frame.pack(expand=True, fill="both", padx=15, pady=20)

        # Tiêu đề mục tác giả
        lbl_author_title = ctk.CTkLabel(
            author_frame, 
            text="TÁC GIẢ THỰC HIỆN", 
            font=ctk.CTkFont(size=14, weight="bold"), 
            text_color="gray75"
        )
        lbl_author_title.pack(pady=(15, 15))

        # Ảnh tác giả tacgia.png
        img_tacgia = self._load_image("tacgia.png", size=(200, 260))
        if img_tacgia:
            lbl_tacgia = ctk.CTkLabel(author_frame, image=img_tacgia, text="")
        else:
            lbl_tacgia = ctk.CTkLabel(
                author_frame, 
                text="[Ảnh: tacgia.png]", 
                width=200, 
                height=260, 
                fg_color="#2B2B2B", 
                corner_radius=8
            )
        lbl_tacgia.pack(pady=5)

        # Tên tác giả
        lbl_name = ctk.CTkLabel(
            author_frame, 
            text="Hàn Quốc Bảo", 
            font=ctk.CTkFont(size=20, weight="bold"), 
            text_color="#1E90FF"
        )
        lbl_name.pack(pady=(15, 2))

        # Trường Đại học Lạc Hồng
        lbl_school = ctk.CTkLabel(
            author_frame, 
            text="Đại học Lạc Hồng", 
            font=ctk.CTkFont(size=15, weight="bold"), 
            text_color="#FFB300"
        )
        lbl_school.pack(pady=(0, 10))

    def _browse_src(self):
        path = filedialog.askdirectory(title="Chọn thư mục done_labels")
        if path:
            self.entry_src.delete(0, "end")
            self.entry_src.insert(0, path)

    def _browse_dest(self):
        path = filedialog.askdirectory(title="Chọn nơi lưu train_yolo_data")
        if path:
            self.entry_dest.delete(0, "end")
            self.entry_dest.insert(0, path)

    def _update_ratio_labels(self, _=None):
        r_train = int(self.slider_train.get())
        r_val = int(self.slider_val.get())
        r_test = int(self.slider_test.get())

        self.lbl_train.configure(text=f"{r_train}%")
        self.lbl_val.configure(text=f"{r_val}%")
        self.lbl_test.configure(text=f"{r_test}%")

        total = r_train + r_val + r_test
        if total == 100:
            self.lbl_total_ratio.configure(text=f"Tổng: {total}% (Hợp lệ)", text_color="#4CAF50")
        else:
            self.lbl_total_ratio.configure(text=f"Tổng: {total}% (Yêu cầu bằng 100%)", text_color="#F44336")

    def _log(self, text: str):
        self.txt_log.insert("end", text + "\n")
        self.txt_log.see("end")

    def _start_split_thread(self):
        if self.is_processing:
            return

        src_dir = self.entry_src.get().strip()
        dest_dir = self.entry_dest.get().strip()

        if not os.path.isdir(src_dir):
            messagebox.showerror("Lỗi", "Thư mục nguồn không hợp lệ!")
            return

        img_dir = os.path.join(src_dir, "images")
        lbl_dir = os.path.join(src_dir, "labels")
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            messagebox.showerror("Lỗi", "Thư mục nguồn phải chứa cả 2 thư mục con: 'images' và 'labels'!")
            return

        if not os.path.isdir(dest_dir):
            messagebox.showerror("Lỗi", "Thư mục đích không hợp lệ!")
            return

        r_train = int(self.slider_train.get())
        r_val = int(self.slider_val.get())
        r_test = int(self.slider_test.get())

        if r_train + r_val + r_test != 100:
            messagebox.showerror("Lỗi tỷ lệ", f"Tổng tỷ lệ hiện tại là {r_train + r_val + r_test}%. Vui lòng điều chỉnh lại đúng 100%!")
            return

        self.is_processing = True
        self.btn_run.configure(state="disabled")
        self.progress_bar.set(0)
        self.txt_log.delete("1.0", "end")

        thread = threading.Thread(
            target=self._process_splitting,
            args=(src_dir, dest_dir, r_train, r_val, r_test, self.chk_shuffle.get() == 1),
            daemon=True
        )
        thread.start()

    def _process_splitting(self, src_dir, dest_dir, r_train, r_val, r_test, is_shuffle):
        try:
            img_dir = os.path.join(src_dir, "images")
            lbl_dir = os.path.join(src_dir, "labels")

            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            all_images = [
                f for f in os.listdir(img_dir) 
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]

            paired_files = []
            for img_file in all_images:
                base_name = os.path.splitext(img_file)[0]
                lbl_file = f"{base_name}.txt"
                lbl_path = os.path.join(lbl_dir, lbl_file)
                if os.path.exists(lbl_path):
                    paired_files.append((img_file, lbl_file))
                else:
                    self._log(f"[!] Cảnh báo: Không tìm thấy label cho ảnh {img_file}. Bỏ qua.")

            total_samples = len(paired_files)
            if total_samples == 0:
                self._log("[x] Không tìm thấy cặp ảnh và nhãn hợp lệ nào.")
                self._finish_task()
                return

            self._log(f"[+] Tìm thấy {total_samples} mẫu dữ liệu hợp lệ.")

            if is_shuffle:
                random.seed(42)
                random.shuffle(paired_files)

            n_train = int(total_samples * (r_train / 100.0))
            n_val = int(total_samples * (r_val / 100.0))
            n_test = total_samples - n_train - n_val

            splits = {
                'train': paired_files[:n_train],
                'valid': paired_files[n_train:n_train + n_val],
                'test': paired_files[n_train + n_val:]
            }

            self._log(f"[+] Phân chia: Train = {len(splits['train'])}, Valid = {len(splits['valid'])}, Test = {len(splits['test'])}")

            target_root = os.path.join(dest_dir, "train_yolo_data")
            for split_name in ['train', 'valid', 'test']:
                os.makedirs(os.path.join(target_root, split_name, "images"), exist_ok=True)
                os.makedirs(os.path.join(target_root, split_name, "labels"), exist_ok=True)

            copied_count = 0
            for split_name, files in splits.items():
                split_img_dir = os.path.join(target_root, split_name, "images")
                split_lbl_dir = os.path.join(target_root, split_name, "labels")

                for img_file, lbl_file in files:
                    shutil.copy2(os.path.join(img_dir, img_file), os.path.join(split_img_dir, img_file))
                    shutil.copy2(os.path.join(lbl_dir, lbl_file), os.path.join(split_lbl_dir, lbl_file))

                    copied_count += 1
                    progress = copied_count / total_samples
                    self.progress_bar.set(progress)
                    self.lbl_status.configure(text=f"Đang sao chép ({split_name}): {copied_count}/{total_samples}")

            src_yaml = os.path.join(src_dir, "data.yaml")
            nc = 0
            names = []

            if os.path.exists(src_yaml):
                with open(src_yaml, "r", encoding="utf-8") as yf:
                    loaded_data = yaml.safe_load(yf)
                    if loaded_data:
                        nc = loaded_data.get("nc", 0)
                        names = loaded_data.get("names", [])

            new_yaml_data = {
                "train": "./train/images",
                "val": "./valid/images",
                "test": "./test/images",
                "nc": nc if nc > 0 else len(names),
                "names": names
            }

            target_yaml = os.path.join(target_root, "data.yaml")
            with open(target_yaml, "w", encoding="utf-8") as yf:
                yaml.dump(new_yaml_data, yf, default_flow_style=None, sort_keys=False)

            self._log(f"[✓] Đã tạo file cấu hình: {target_yaml}")
            self._log(f"[✓] Đã tạo thành công bộ dữ liệu YOLO tại: {target_root}")
            messagebox.showinfo("Thành công", f"Chia dữ liệu thành công!\nThư mục: {target_root}")

        except Exception as e:
            self._log(f"[x] Lỗi: {str(e)}")
            messagebox.showerror("Lỗi hệ thống", f"Có lỗi xảy ra: {str(e)}")
        finally:
            self._finish_task()

    def _finish_task(self):
        self.is_processing = False
        self.btn_run.configure(state="normal")
        self.lbl_status.configure(text="Trạng thái: Hoàn tất")


if __name__ == "__main__":
    app = YOLODatasetSplitterApp()
    app.mainloop()