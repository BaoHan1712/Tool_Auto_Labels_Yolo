import os
import shutil
import random
import yaml
import threading
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# -----------------------------------------------------------------
# ENGINE XỬ LÝ TỪNG LOẠI TĂNG CƯỜNG ĐỘC LẬP & TÍNH TOÁN BBOX YOLO
# -----------------------------------------------------------------
class SingleAugmentationEngine:
    @staticmethod
    def yolo_to_voc(box, w, h):
        cls_id, xc, yc, bw, bh = box
        x1 = int((xc - bw / 2.0) * w)
        y1 = int((yc - bh / 2.0) * h)
        x2 = int((xc + bw / 2.0) * w)
        y2 = int((yc + bh / 2.0) * h)
        return cls_id, max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    @staticmethod
    def voc_to_yolo(cls_id, x1, y1, x2, y2, w, h):
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0
        return [cls_id, xc / w, yc / h, bw / w, bh / h]

    @classmethod
    def apply_single(cls, aug_type, img, boxes, intensity, is_preview=False, variant=None):
        if intensity <= 0:
            return img.copy(), [list(b) for b in boxes]

        h, w = img.shape[:2]
        res_img = img.copy()
        res_boxes = [list(b) for b in boxes]

        if aug_type == "blur":
            k_size = int(intensity * 20) // 2 * 2 + 3
            res_img = cv2.GaussianBlur(img, (k_size, k_size), 0)

        elif aug_type == "motion_blur":
            k_size = int(intensity * 24) // 2 * 2 + 3
            kernel = np.zeros((k_size, k_size))
            kernel[int((k_size - 1) / 2), :] = np.ones(k_size) / k_size
            res_img = cv2.filter2D(img, -1, kernel)

        elif aug_type == "saturation":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            brightness_factor = max(0.2, 1.0 - intensity) if variant == "dark" else 1.0 + intensity
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
            res_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        elif aug_type == "rotation":
            max_angle = intensity * 35.0
            if variant == "left":
                angle = max_angle
            elif variant == "right":
                angle = -max_angle
            else:
                angle = 0
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            res_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            new_boxes = []
            for b in boxes:
                cls_id, x1, y1, x2, y2 = cls.yolo_to_voc(b, w, h)
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                pts_rot = cv2.transform(np.array([pts]), M)[0]
                rx1 = max(0, int(np.min(pts_rot[:, 0])))
                ry1 = max(0, int(np.min(pts_rot[:, 1])))
                rx2 = min(w, int(np.max(pts_rot[:, 0])))
                ry2 = min(h, int(np.max(pts_rot[:, 1])))
                if (rx2 - rx1) > 4 and (ry2 - ry1) > 4:
                    new_boxes.append(cls.voc_to_yolo(cls_id, rx1, ry1, rx2, ry2, w, h))
            res_boxes = new_boxes

        elif aug_type == "crop":
            crop_ratio = 1.0 - (intensity * 0.35)
            nw, nh = int(w * crop_ratio), int(h * crop_ratio)
            
            # Preview thì cắt ở tâm, chạy Batch thực tế thì cắt ngẫu nhiên các góc
            if is_preview:
                cx = (w - nw) // 2
                cy = (h - nh) // 2
            else:
                cx = random.randint(0, max(0, w - nw))
                cy = random.randint(0, max(0, h - nh))

            cropped = img[cy:cy + nh, cx:cx + nw]
            res_img = cv2.resize(cropped, (w, h))

            new_boxes = []
            for b in boxes:
                cls_id, x1, y1, x2, y2 = cls.yolo_to_voc(b, w, h)
                nx1 = max(0, x1 - cx)
                ny1 = max(0, y1 - cy)
                nx2 = min(nw, x2 - cx)
                ny2 = min(nh, y2 - cy)
                if nx2 > nx1 and ny2 > ny1:
                    scale_x, scale_y = w / nw, h / nh
                    fx1 = int(nx1 * scale_x)
                    fy1 = int(ny1 * scale_y)
                    fx2 = int(nx2 * scale_x)
                    fy2 = int(ny2 * scale_y)
                    if (fx2 - fx1) > 5 and (fy2 - fy1) > 5:
                        new_boxes.append(cls.voc_to_yolo(cls_id, fx1, fy1, fx2, fy2, w, h))
            res_boxes = new_boxes

        return res_img, res_boxes


# -----------------------------------------------------------------
# CỬA SỔ POPUP CẤU HÌNH CHO TỪNG LOẠI TĂNG CƯỜNG (APPLY / CANCEL)
# -----------------------------------------------------------------
class AugmentationModal(ctk.CTkToplevel):
    def __init__(self, parent, aug_key, aug_name, orig_img, orig_boxes, class_names, current_val, on_apply_callback):
        super().__init__(parent)
        self.title(f"Tùy chỉnh: {aug_name}")
        self.geometry("640x590")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.aug_key = aug_key
        self.orig_img = orig_img
        self.orig_boxes = orig_boxes
        self.class_names = class_names
        self.on_apply_callback = on_apply_callback
        self.temp_intensity = current_val

        self._setup_ui(aug_name)
        self._update_views()

    def _setup_ui(self, aug_name):
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=15, pady=(10, 5))

        ctk.CTkLabel(top, text=f"Tùy biến mức độ: {aug_name}", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=10, pady=(6, 2))

        slider_row = ctk.CTkFrame(top, fg_color="transparent")
        slider_row.pack(fill="x", padx=10, pady=5)

        self.slider = ctk.CTkSlider(slider_row, from_=0.05, to=1.0, number_of_steps=19, command=self._on_slider_change)
        self.slider.set(self.temp_intensity if self.temp_intensity > 0 else 0.3)
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.lbl_percent = ctk.CTkLabel(slider_row, text=f"{int(self.slider.get() * 100)}%", width=45, font=ctk.CTkFont(weight="bold"))
        self.lbl_percent.pack(side="right")

        preview_box = ctk.CTkFrame(self)
        preview_box.pack(fill="both", expand=True, padx=15, pady=5)

        ctk.CTkLabel(preview_box, text="Ảnh gốc (Original Image)", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=8, pady=(4, 0))
        self.lbl_orig = ctk.CTkLabel(preview_box, text="", fg_color="#181818", corner_radius=6, height=190)
        self.lbl_orig.pack(fill="x", padx=8, pady=(2, 6))

        ctk.CTkLabel(preview_box, text=f"Kết quả sau khi chỉnh sửa ({aug_name})", font=ctk.CTkFont(size=11, weight="bold"), text_color="#64B5F6").pack(anchor="w", padx=8, pady=(4, 0))
        self.result_frame = ctk.CTkFrame(preview_box, fg_color="transparent")
        self.result_frame.pack(fill="x", padx=8, pady=(2, 6))
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_columnconfigure(1, weight=1)

        self.lbl_aug_left = ctk.CTkLabel(self.result_frame, text="", fg_color="#181818", corner_radius=6, height=145)
        self.lbl_aug_left.grid(row=0, column=0, padx=(0, 4), sticky="nsew")
        self.lbl_aug_right = ctk.CTkLabel(self.result_frame, text="", fg_color="#181818", corner_radius=6, height=145)
        self.lbl_aug_right.grid(row=0, column=1, padx=(4, 0), sticky="nsew")

        btn_bar = ctk.CTkFrame(self, fg_color="transparent")
        btn_bar.pack(fill="x", padx=15, pady=8)

        self.btn_cancel = ctk.CTkButton(btn_bar, text="Cancel (Thoát ra)", fg_color="#B71C1C", hover_color="#7F0000", width=130, command=self.destroy)
        self.btn_cancel.pack(side="left", padx=5)

        self.btn_apply = ctk.CTkButton(btn_bar, text="Apply (Xác nhận chọn)", fg_color="#2E7D32", hover_color="#1B5E20", width=170, font=ctk.CTkFont(weight="bold"), command=self._apply)
        self.btn_apply.pack(side="right", padx=5)

    def _on_slider_change(self, val):
        self.temp_intensity = val
        self.lbl_percent.configure(text=f"{int(val * 100)}%")
        self._update_views()

    def _draw_boxes(self, img_bgr, boxes):
        canvas = img_bgr.copy()
        h, w = canvas.shape[:2]
        colors = [(0, 255, 100), (0, 150, 255), (255, 100, 0), (255, 0, 255)]
        for b in boxes:
            cls_id, xc, yc, bw, bh = b
            cls_id = int(cls_id)
            color = colors[cls_id % len(colors)]
            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            c_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            cv2.putText(canvas, c_name, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return canvas

    def _render_to_widget(self, cv_img, boxes, widget, max_width=580, max_height=185):
        rendered = self._draw_boxes(cv_img, boxes)
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]
        ratio = min(max_width / iw, max_height / ih)
        nw, nh = int(iw * ratio), int(ih * ratio)
        pil_img = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(nw, nh))
        widget.configure(image=ctk_img)
        widget.image = ctk_img

    def _update_views(self):
        if self.orig_img is None:
            return
        self._render_to_widget(self.orig_img, self.orig_boxes, self.lbl_orig)

        variants = {
            "rotation": (("Xoay trái", "left"), ("Xoay phải", "right")),
            "saturation": (("Tối", "dark"), ("Sáng", "bright"))
        }.get(self.aug_key)

        if variants:
            for widget, (label, variant) in zip(
                (self.lbl_aug_left, self.lbl_aug_right), variants
            ):
                aug_img, aug_boxes = SingleAugmentationEngine.apply_single(
                    self.aug_key, self.orig_img, self.orig_boxes,
                    self.temp_intensity, is_preview=True, variant=variant
                )
                widget.configure(text=label)
                self._render_to_widget(aug_img, aug_boxes, widget, max_width=280, max_height=140)
        else:
            aug_img, aug_boxes = SingleAugmentationEngine.apply_single(
                self.aug_key, self.orig_img, self.orig_boxes, self.temp_intensity, is_preview=True
            )
            self.lbl_aug_left.configure(text="Kết quả")
            self._render_to_widget(aug_img, aug_boxes, self.lbl_aug_left, max_width=580, max_height=145)
            self.lbl_aug_right.configure(image=None, text="")

    def _apply(self):
        self.on_apply_callback(self.aug_key, self.temp_intensity)
        self.destroy()


# -----------------------------------------------------------------
# GIAO DIỆN CHÍNH (ĐƯỢC THIẾT KẾ GỌN GÀNG, CAO 620PX)
# -----------------------------------------------------------------
class YOLOAugmentationStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Roboflow-style YOLO Augmentation Studio")
        self.geometry("820x620")
        self.resizable(False, False)

        self.data_dir = ""
        self.class_names = []
        self.orig_cv_img = None
        self.orig_boxes = []
        self.sample_filename = ""
        self.is_processing = False

        self.applied_configs = {
            "crop": 0.0,
            "rotation": 0.0,
            "saturation": 0.0,
            "motion_blur": 0.0,
            "blur": 0.0
        }

        self.aug_metadata = [
            ("crop", "Crop (Cắt xén / Zoom)", "Cắt cúp ngẫu nhiên và tự động resize lại bbox"),
            ("rotation", "Rotation (Xoay góc)", "Xoay ảnh và tính toán lại toạ độ xoay của bbox"),
            ("saturation", "Saturation (Màu sắc / Tối - sáng)", "Điều chỉnh độ rực màu và tạo ảnh tối hoặc sáng"),
            ("motion_blur", "Motion Blur (Làm mờ chuyển động)", "Giả lập hiện tượng camera rung hoặc vật thể chuyển động"),
            ("blur", "Gaussian Blur (Làm mờ nét)", "Làm mờ đều các chi tiết trên bề mặt ảnh")
        ]

        self.status_labels = {}
        self._setup_ui()

    def _setup_ui(self):
        top_bar = ctk.CTkFrame(self)
        top_bar.pack(fill="x", padx=15, pady=(10, 5))

        ctk.CTkLabel(top_bar, text="done_labels:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.entry_dir = ctk.CTkEntry(top_bar, width=440, placeholder_text="Chọn thư mục chứa done_labels...")
        self.entry_dir.grid(row=0, column=1, padx=5, pady=8)
        ctk.CTkButton(top_bar, text="Duyệt...", width=80, command=self._browse_dir).grid(row=0, column=2, padx=5, pady=8)
        ctk.CTkButton(top_bar, text="Đổi ảnh mẫu", width=95, fg_color="#37474F", command=self._pick_sample).grid(row=0, column=3, padx=5, pady=8)

        list_frame = ctk.CTkScrollableFrame(self, height=330, label_text="DANH SÁCH CÁC BỘ TĂNG CƯỜNG (CHỌN TỪNG LOẠI ĐỂ CẤU HÌNH)")
        list_frame.pack(fill="x", padx=15, pady=5)

        for key, name, desc in self.aug_metadata:
            card = ctk.CTkFrame(list_frame)
            card.pack(fill="x", padx=5, pady=4)

            info_col = ctk.CTkFrame(card, fg_color="transparent")
            info_col.pack(side="left", fill="both", expand=True, padx=10, pady=5)
            ctk.CTkLabel(info_col, text=name, font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w")
            ctk.CTkLabel(info_col, text=desc, font=ctk.CTkFont(size=11), text_color="gray70").pack(anchor="w")

            status_lbl = ctk.CTkLabel(card, text="Trạng thái: Chưa chọn (0%)", width=180, font=ctk.CTkFont(size=12))
            status_lbl.pack(side="left", padx=10)
            self.status_labels[key] = status_lbl

            btn_cfg = ctk.CTkButton(card, text="Tùy biến %", width=100, command=lambda k=key, n=name: self._open_modal(k, n))
            btn_cfg.pack(side="right", padx=10, pady=8)

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(bottom_frame, text="Nhân số lượng ảnh lên x").pack(side="left", padx=(15, 5), pady=8)
        self.cmb_mult = ctk.CTkComboBox(bottom_frame, values=["1", "2", "3", "5"], width=75)
        self.cmb_mult.set("2")
        self.cmb_mult.pack(side="left", padx=5)

        self.lbl_selected_summary = ctk.CTkLabel(bottom_frame, text="Đã chọn: 0 loại", text_color="#64B5F6", font=ctk.CTkFont(weight="bold"))
        self.lbl_selected_summary.pack(side="left", padx=15)

        self.btn_run = ctk.CTkButton(bottom_frame, text="TẠO TOÀN BỘ DATASET", fg_color="#1E88E5", hover_color="#1565C0",
                                     font=ctk.CTkFont(size=13, weight="bold"), width=190, command=self._start_batch_thread)
        self.btn_run.pack(side="right", padx=10, pady=8)

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.pack(fill="x", padx=15, pady=(5, 2))
        self.progress_bar.set(0)

        self.lbl_status = ctk.CTkLabel(self, text="Vui lòng chọn thư mục done_labels để bắt đầu", text_color="gray70", font=ctk.CTkFont(size=11))
        self.lbl_status.pack(anchor="w", padx=15, pady=(0, 5))

    def _browse_dir(self):
        path = filedialog.askdirectory(title="Chọn thư mục done_labels")
        if path:
            self.data_dir = path
            self.entry_dir.delete(0, "end")
            self.entry_dir.insert(0, path)
            self._load_yaml()
            self._pick_sample()

    def _load_yaml(self):
        yaml_path = os.path.join(self.data_dir, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
                if d and "names" in d:
                    self.class_names = d["names"]

    def _pick_sample(self):
        if not self.data_dir:
            return
        img_dir = os.path.join(self.data_dir, "images")
        lbl_dir = os.path.join(self.data_dir, "labels")
        if not os.path.isdir(img_dir):
            messagebox.showerror("Lỗi", "Không tìm thấy folder 'images' trong thư mục đã chọn!")
            return

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        imgs = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        if not imgs:
            messagebox.showwarning("Cảnh báo", "Không có ảnh hợp lệ nào!")
            return

        self.sample_filename = random.choice(imgs)
        self.orig_cv_img = cv2.imread(os.path.join(img_dir, self.sample_filename))
        self.orig_boxes = []

        lbl_path = os.path.join(lbl_dir, f"{os.path.splitext(self.sample_filename)[0]}.txt")
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) == 5:
                        self.orig_boxes.append([int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])

        self.lbl_status.configure(text=f"Đang dùng ảnh mẫu: {self.sample_filename}")

    def _open_modal(self, key, name):
        if self.orig_cv_img is None:
            messagebox.showwarning("Chưa có ảnh", "Vui lòng chọn thư mục done_labels có chứa ảnh trước!")
            return

        current_val = self.applied_configs.get(key, 0.0)
        AugmentationModal(
            self, key, name, self.orig_cv_img, self.orig_boxes, self.class_names, current_val, self._on_apply_from_modal
        )

    def _on_apply_from_modal(self, key, intensity):
        self.applied_configs[key] = intensity
        pct = int(intensity * 100)
        self.status_labels[key].configure(
            text=f"✔ Đã bật ({pct}%)",
            text_color="#4CAF50"
        )
        active_count = sum(1 for v in self.applied_configs.values() if v > 0)
        self.lbl_selected_summary.configure(text=f"Đã chọn: {active_count} loại")

    def _start_batch_thread(self):
        if self.is_processing:
            return

        if not self.data_dir or not os.path.isdir(self.data_dir):
            messagebox.showerror("Lỗi", "Chưa chọn thư mục dữ liệu!")
            return

        active_augs = {k: v for k, v in self.applied_configs.items() if v > 0}
        if not active_augs:
            messagebox.showwarning("Chưa có loại nào", "Bạn chưa bấm Apply cho loại tăng cường nào cả!")
            return

        self.is_processing = True
        self.btn_run.configure(state="disabled")
        self.progress_bar.set(0)

        mult = int(self.cmb_mult.get())
        t = threading.Thread(target=self._run_batch, args=(active_augs, mult), daemon=True)
        t.start()

    def _run_batch(self, active_augs, mult):
        try:
            parent = os.path.dirname(os.path.abspath(self.data_dir))
            out_root = os.path.join(parent, "done_labels_augmented")
            out_img = os.path.join(out_root, "images")
            out_lbl = os.path.join(out_root, "labels")
            os.makedirs(out_img, exist_ok=True)
            os.makedirs(out_lbl, exist_ok=True)

            src_img = os.path.join(self.data_dir, "images")
            src_lbl = os.path.join(self.data_dir, "labels")

            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            files = [f for f in os.listdir(src_img) if os.path.splitext(f)[1].lower() in valid_exts]

            variant_names = {
                "rotation": ("left", "right"),
                "saturation": ("dark", "bright")
            }
            total_per_file = 1 + sum(
                mult * len(variant_names.get(aug_key, (None,)))
                for aug_key in active_augs
            )
            total = len(files) * total_per_file
            cnt = 0

            for f in files:
                bname, ext = os.path.splitext(f)
                im_path = os.path.join(src_img, f)
                lb_path = os.path.join(src_lbl, f"{bname}.txt")

                img = cv2.imread(im_path)
                if img is None:
                    continue

                boxes = []
                if os.path.exists(lb_path):
                    with open(lb_path, "r", encoding="utf-8") as lf:
                        for line in lf:
                            p = line.strip().split()
                            if len(p) == 5:
                                boxes.append([int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])

                # 1. Lưu ảnh gốc và nhãn gốc
                shutil.copy2(im_path, os.path.join(out_img, f))
                if os.path.exists(lb_path):
                    shutil.copy2(lb_path, os.path.join(out_lbl, f"{bname}.txt"))
                cnt += 1
                self.progress_bar.set(cnt / total)

                # 2. Sinh ảnh tăng cường: DUYỆT QUA TỪNG LOẠI ĐÃ APPLY, MỖI LOẠI RA 'mult' ẢNH
                for aug_key, intensity in active_augs.items():
                    variants = variant_names.get(aug_key, (None,))
                    for i in range(1, mult + 1):
                        for variant in variants:
                            aug_im, aug_bx = SingleAugmentationEngine.apply_single(
                                aug_key, img, boxes, intensity, is_preview=False, variant=variant
                            )

                            suffix = f"_{variant}" if variant else ""
                            out_name = f"{bname}_{aug_key}{suffix}_{i}{ext}"
                            out_txt = f"{bname}_{aug_key}{suffix}_{i}.txt"

                            cv2.imwrite(os.path.join(out_img, out_name), aug_im)
                            with open(os.path.join(out_lbl, out_txt), "w", encoding="utf-8") as lf:
                                for b in aug_bx:
                                    lf.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                            cnt += 1
                            self.progress_bar.set(cnt / total)
                            self.lbl_status.configure(text=f"Đang sinh: {out_name}")

            # Copy data.yaml
            src_yaml = os.path.join(self.data_dir, "data.yaml")
            if os.path.exists(src_yaml):
                shutil.copy2(src_yaml, os.path.join(out_root, "data.yaml"))

            self.lbl_status.configure(text="Đã hoàn tất toàn bộ quá trình tăng cường dữ liệu!")
            messagebox.showinfo("Hoàn thành", f"Dữ liệu đã được tạo thành công tại:\n{out_root}")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {str(e)}")
        finally:
            self.is_processing = False
            self.btn_run.configure(state="normal")


if __name__ == "__main__":
    app = YOLOAugmentationStudioApp()
    app.mainloop()