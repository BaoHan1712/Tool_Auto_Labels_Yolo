import os
import shutil
import random
import threading
import yaml
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
from PIL import Image, ImageTk, ImageDraw

# Cấu hình giao diện CustomTkinter tông màu xanh biển
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Theme Palette: Xanh biển
COLOR_BG_SIDEBAR = "#0F1A2C"       # Xanh navy đậm
COLOR_BG_MAIN = "#152238"          # Xanh biển đêm
COLOR_CARD = "#1C2D4A"             # Card container xanh navy
COLOR_ACCENT = "#1976D2"           # Xanh biển sáng
COLOR_ACCENT_HOVER = "#1565C0"     # Xanh biển hover
COLOR_ACCENT_LIGHT = "#64B5F6"     # Xanh da trời nhạt (Text & Highlight)
COLOR_TEXT_MAIN = "#E3F2FD"


# =================================================================
# 1. LOGIC ENGINE MODULE TĂNG CƯỜNG DỮ LIỆU (FILE 2)
# =================================================================
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
            angle = max_angle if variant == "left" else (-max_angle if variant == "right" else 0)
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
        top = ctk.CTkFrame(self, fg_color=COLOR_CARD)
        top.pack(fill="x", padx=15, pady=(10, 5))

        ctk.CTkLabel(top, text=f"Tùy biến mức độ: {aug_name}", font=ctk.CTkFont(size=15, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=10, pady=(6, 2))

        slider_row = ctk.CTkFrame(top, fg_color="transparent")
        slider_row.pack(fill="x", padx=10, pady=5)

        self.slider = ctk.CTkSlider(slider_row, from_=0.05, to=1.0, number_of_steps=19, progress_color=COLOR_ACCENT, command=self._on_slider_change)
        self.slider.set(self.temp_intensity if self.temp_intensity > 0 else 0.3)
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.lbl_percent = ctk.CTkLabel(slider_row, text=f"{int(self.slider.get() * 100)}%", width=45, font=ctk.CTkFont(weight="bold"), text_color=COLOR_ACCENT_LIGHT)
        self.lbl_percent.pack(side="right")

        preview_box = ctk.CTkFrame(self, fg_color=COLOR_CARD)
        preview_box.pack(fill="both", expand=True, padx=15, pady=5)

        ctk.CTkLabel(preview_box, text="Ảnh gốc (Original Image)", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=8, pady=(4, 0))
        self.lbl_orig = ctk.CTkLabel(preview_box, text="", fg_color="#101A28", corner_radius=6, height=185)
        self.lbl_orig.pack(fill="x", padx=8, pady=(2, 6))

        ctk.CTkLabel(preview_box, text=f"Kết quả mô phỏng ({aug_name})", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_ACCENT_LIGHT).pack(anchor="w", padx=8, pady=(4, 0))
        self.result_frame = ctk.CTkFrame(preview_box, fg_color="transparent")
        self.result_frame.pack(fill="x", padx=8, pady=(2, 6))
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_columnconfigure(1, weight=1)

        self.lbl_aug_left = ctk.CTkLabel(self.result_frame, text="", fg_color="#101A28", corner_radius=6, height=140)
        self.lbl_aug_left.grid(row=0, column=0, padx=(0, 4), sticky="nsew")
        self.lbl_aug_right = ctk.CTkLabel(self.result_frame, text="", fg_color="#101A28", corner_radius=6, height=140)
        self.lbl_aug_right.grid(row=0, column=1, padx=(4, 0), sticky="nsew")

        btn_bar = ctk.CTkFrame(self, fg_color="transparent")
        btn_bar.pack(fill="x", padx=15, pady=8)

        self.btn_cancel = ctk.CTkButton(btn_bar, text="Thoát", fg_color="#D32F2F", hover_color="#9A0007", width=120, command=self.destroy)
        self.btn_cancel.pack(side="left", padx=5)

        self.btn_apply = ctk.CTkButton(btn_bar, text="Xác nhận áp dụng", fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, width=160, font=ctk.CTkFont(weight="bold"), command=self._apply)
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

    def _render_to_widget(self, cv_img, boxes, widget, max_width=580, max_height=180):
        rendered = self._draw_boxes(cv_img, boxes)
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]
        ratio = min(max_width / iw, max_height / ih)
        nw, nh = max(1, int(iw * ratio)), max(1, int(ih * ratio))
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
            for widget, (label, variant) in zip((self.lbl_aug_left, self.lbl_aug_right), variants):
                aug_img, aug_boxes = SingleAugmentationEngine.apply_single(
                    self.aug_key, self.orig_img, self.orig_boxes, self.temp_intensity, is_preview=True, variant=variant
                )
                widget.configure(text=label)
                self._render_to_widget(aug_img, aug_boxes, widget, max_width=280, max_height=135)
        else:
            aug_img, aug_boxes = SingleAugmentationEngine.apply_single(
                self.aug_key, self.orig_img, self.orig_boxes, self.temp_intensity, is_preview=True
            )
            self.lbl_aug_left.configure(text="Kết quả")
            self._render_to_widget(aug_img, aug_boxes, self.lbl_aug_left, max_width=580, max_height=135)
            self.lbl_aug_right.configure(image=None, text="")

    def _apply(self):
        self.on_apply_callback(self.aug_key, self.temp_intensity)
        self.destroy()


# =================================================================
# 2. POPUP REVIEW & CHỈNH SỬA BBOX SAU KHI GÁN NHÃN (FILE 1)
# =================================================================
class AnnotationEditor(ctk.CTkToplevel):
    def __init__(self, parent, done_dir, class_names):
        super().__init__(parent)
        self.title("YOLO Annotation Editor & Reviewer")
        self.geometry("1100x750")
        
        self.done_dir = done_dir
        self.images_dir = os.path.join(done_dir, "images")
        self.labels_dir = os.path.join(done_dir, "labels")
        self.class_names = class_names
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]) if os.path.exists(self.images_dir) else []
        
        self.current_idx = 0
        self.pending_annotations = {}
        
        self.orig_img = None
        self.tk_img = None
        self.img_w, self.img_h = 0, 0
        self.scale_x, self.scale_y = 1.0, 1.0
        self.offset_x, self.offset_y = 0, 0
        
        self.boxes = []
        self.selected_box_idx = None
        self.drag_mode = None
        self.active_handle = None
        self.drag_start_x, self.drag_start_y = 0, 0
        self.HANDLE_SIZE = 7
        self.colors = ["#00E676", "#FF1744", "#2979FF", "#FFEA00", "#D500F9", "#00E5FF"]
        
        self._setup_ui()
        self._load_current_image()

    def _setup_ui(self):
        top_bar = ctk.CTkFrame(self, fg_color=COLOR_CARD)
        top_bar.pack(fill="x", padx=10, pady=5)
        
        self.btn_prev = ctk.CTkButton(top_bar, text="◀ Trước", width=90, fg_color=COLOR_ACCENT, command=self._prev_image)
        self.btn_prev.pack(side="left", padx=5, pady=5)
        
        self.lbl_counter = ctk.CTkLabel(top_bar, text="0 / 0", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_counter.pack(side="left", padx=10)
        
        self.btn_next = ctk.CTkButton(top_bar, text="Sau ▶", width=90, fg_color=COLOR_ACCENT, command=self._next_image)
        self.btn_next.pack(side="left", padx=5, pady=5)
        
        ctk.CTkLabel(top_bar, text="Class vẽ:").pack(side="left", padx=(15, 5))
        self.cmb_class = ctk.CTkComboBox(top_bar, values=self.class_names, width=130)
        if self.class_names:
            self.cmb_class.set(self.class_names[0])
        self.cmb_class.pack(side="left", padx=5)
        
        self.btn_del = ctk.CTkButton(top_bar, text="Xóa Box (Del)", fg_color="#D32F2F", hover_color="#9A0007", width=100, command=self._delete_selected_box)
        self.btn_del.pack(side="left", padx=10)
        
        self.btn_save = ctk.CTkButton(top_bar, text="💾 Lưu tất cả", fg_color="#2E7D32", hover_color="#1B5E20", width=130, command=self._save_all_annotations)
        self.btn_save.pack(side="right", padx=10)

        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.canvas = Canvas(self.canvas_frame, bg="#0E1624", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.bind("<Delete>", lambda e: self._delete_selected_box())
        self.bind("<BackSpace>", lambda e: self._delete_selected_box())

    def _load_current_image(self):
        if not self.image_files:
            return
            
        file_name = self.image_files[self.current_idx]
        self.lbl_counter.configure(text=f"[{self.current_idx + 1}/{len(self.image_files)}] {file_name}")
        
        img_path = os.path.join(self.images_dir, file_name)
        self.orig_img = Image.open(img_path)
        self.img_w, self.img_h = self.orig_img.size
        
        base_name = os.path.splitext(file_name)[0]
        label_path = os.path.join(self.labels_dir, f"{base_name}.txt")
        
        raw_boxes = self.pending_annotations.get(file_name)
        if raw_boxes is None and os.path.exists(label_path):
            raw_boxes = []
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        x1 = xc - w / 2.0
                        y1 = yc - h / 2.0
                        x2 = xc + w / 2.0
                        y2 = yc + h / 2.0
                        raw_boxes.append({'cls': cls_id, 'norm_coords': [x1, y1, x2, y2]})
        
        self.raw_boxes = [
            {'cls': item['cls'], 'norm_coords': item['norm_coords'][:]}
            for item in (raw_boxes or [])
        ]
        self.selected_box_idx = None
        self.update_idletasks()
        self._render_canvas()

    def _render_canvas(self):
        self.canvas.delete("all")
        if self.orig_img is None:
            return

        canv_w = self.canvas.winfo_width()
        canv_h = self.canvas.winfo_height()
        if canv_w <= 10 or canv_h <= 10:
            canv_w, canv_h = 960, 600

        ratio = min(canv_w / self.img_w, canv_h / self.img_h)
        new_w = max(1, int(self.img_w * ratio))
        new_h = max(1, int(self.img_h * ratio))
        
        self.scale_x = new_w
        self.scale_y = new_h
        self.offset_x = (canv_w - new_w) // 2
        self.offset_y = (canv_h - new_h) // 2

        resized = self.orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_img)

        self.boxes = []
        for item in self.raw_boxes:
            nx1, ny1, nx2, ny2 = item['norm_coords']
            x1 = self.offset_x + nx1 * self.scale_x
            y1 = self.offset_y + ny1 * self.scale_y
            x2 = self.offset_x + nx2 * self.scale_x
            y2 = self.offset_y + ny2 * self.scale_y
            self.boxes.append({'cls': item['cls'], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

        self._draw_boxes()

    def _draw_boxes(self):
        self.canvas.delete("box_element")
        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            cls_id = box['cls']
            color = self.colors[cls_id % len(self.colors)]
            is_selected = (idx == self.selected_box_idx)
            outline_w = 3 if is_selected else 2

            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=outline_w, tags="box_element")
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"ID_{cls_id}"
            tag_text = f" {cls_name} "
            self.canvas.create_rectangle(x1, y1 - 18, x1 + len(tag_text) * 8, y1, fill=color, outline=color, tags="box_element")
            self.canvas.create_text(x1 + 2, y1 - 9, text=tag_text, fill="#000000", font=("Arial", 9, "bold"), anchor="w", tags="box_element")

            if is_selected:
                hs = self.HANDLE_SIZE
                for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    self.canvas.create_rectangle(cx - hs, cy - hs, cx + hs, cy + hs, fill="#FFFFFF", outline=color, tags="box_element")

    def _get_hit_handle(self, box, x, y):
        hs = self.HANDLE_SIZE + 4
        handles = {
            'top_left': (box['x1'], box['y1']),
            'top_right': (box['x2'], box['y1']),
            'bottom_left': (box['x1'], box['y2']),
            'bottom_right': (box['x2'], box['y2']),
        }
        for name, (hx, hy) in handles.items():
            if abs(x - hx) <= hs and abs(y - hy) <= hs:
                return name
        return None

    def _on_mouse_down(self, event):
        x, y = event.x, event.y
        self.drag_start_x, self.drag_start_y = x, y

        if self.selected_box_idx is not None:
            active_box = self.boxes[self.selected_box_idx]
            handle = self._get_hit_handle(active_box, x, y)
            if handle:
                self.drag_mode = 'resize'
                self.active_handle = handle
                return

        for idx in reversed(range(len(self.boxes))):
            b = self.boxes[idx]
            if min(b['x1'], b['x2']) <= x <= max(b['x1'], b['x2']) and min(b['y1'], b['y2']) <= y <= max(b['y1'], b['y2']):
                self.selected_box_idx = idx
                self.drag_mode = 'move'
                self._draw_boxes()
                return

        self.selected_box_idx = None
        current_cls = self.cmb_class.get()
        cls_id = self.class_names.index(current_cls) if current_cls in self.class_names else 0
        new_box = {'cls': cls_id, 'x1': x, 'y1': y, 'x2': x, 'y2': y}
        self.boxes.append(new_box)
        self.selected_box_idx = len(self.boxes) - 1
        self.drag_mode = 'create'
        self._draw_boxes()

    def _on_mouse_drag(self, event):
        if self.selected_box_idx is None or self.drag_mode is None:
            return

        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        b = self.boxes[self.selected_box_idx]

        if self.drag_mode == 'move':
            b['x1'] += dx
            b['y1'] += dy
            b['x2'] += dx
            b['y2'] += dy
            self.drag_start_x, self.drag_start_y = event.x, event.y
        elif self.drag_mode == 'resize':
            if 'left' in self.active_handle:
                b['x1'] += dx
            if 'right' in self.active_handle:
                b['x2'] += dx
            if 'top' in self.active_handle:
                b['y1'] += dy
            if 'bottom' in self.active_handle:
                b['y2'] += dy
            self.drag_start_x, self.drag_start_y = event.x, event.y
        elif self.drag_mode == 'create':
            b['x2'], b['y2'] = event.x, event.y

        self._draw_boxes()

    def _on_mouse_up(self, event):
        if self.selected_box_idx is not None:
            b = self.boxes[self.selected_box_idx]
            x1, x2 = min(b['x1'], b['x2']), max(b['x1'], b['x2'])
            y1, y2 = min(b['y1'], b['y2']), max(b['y1'], b['y2'])
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                self.boxes.pop(self.selected_box_idx)
                self.selected_box_idx = None
            else:
                b['x1'], b['x2'], b['y1'], b['y2'] = x1, x2, y1, y2
        self.drag_mode = None
        self.active_handle = None
        self._sync_back_to_raw()
        self._draw_boxes()

    def _sync_back_to_raw(self):
        self.raw_boxes = []
        for b in self.boxes:
            nx1 = max(0.0, min(1.0, (b['x1'] - self.offset_x) / self.scale_x))
            ny1 = max(0.0, min(1.0, (b['y1'] - self.offset_y) / self.scale_y))
            nx2 = max(0.0, min(1.0, (b['x2'] - self.offset_x) / self.scale_x))
            ny2 = max(0.0, min(1.0, (b['y2'] - self.offset_y) / self.scale_y))
            self.raw_boxes.append({'cls': b['cls'], 'norm_coords': [nx1, ny1, nx2, ny2]})

    def _delete_selected_box(self):
        if self.selected_box_idx is not None and 0 <= self.selected_box_idx < len(self.boxes):
            self.boxes.pop(self.selected_box_idx)
            self.selected_box_idx = None
            self._sync_back_to_raw()
            self._draw_boxes()

    def _update_current_annotation(self):
        if not self.image_files:
            return
        self._sync_back_to_raw()
        file_name = self.image_files[self.current_idx]
        self.pending_annotations[file_name] = [
            {'cls': item['cls'], 'norm_coords': item['norm_coords'][:]}
            for item in self.raw_boxes
        ]

    def _save_all_annotations(self):
        if not self.image_files:
            return
        self._update_current_annotation()
        for file_name, raw_boxes in self.pending_annotations.items():
            base_name = os.path.splitext(file_name)[0]
            label_path = os.path.join(self.labels_dir, f"{base_name}.txt")
            lines = []
            for item in raw_boxes:
                cls_id = item['cls']
                x1, y1, x2, y2 = item['norm_coords']
                w, h = abs(x2 - x1), abs(y2 - y1)
                xc, yc = x1 + w / 2.0, y1 + h / 2.0
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            with open(label_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        messagebox.showinfo("Thành công", f"Đã cập nhật nhãn cho {len(self.pending_annotations)} ảnh!", parent=self)

    def _prev_image(self):
        if self.current_idx > 0:
            self._update_current_annotation()
            self.current_idx -= 1
            self._load_current_image()

    def _next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self._update_current_annotation()
            self.current_idx += 1
            self._load_current_image()


# =================================================================
# 3. CÁC VIEW GIAO DIỆN CON (FRAMES ĐẶT TẠI CỘT GIỮA)
# =================================================================

class AutoLabelView(ctk.CTkFrame):
    """View 1: Gán nhãn tự động"""
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.is_processing = False

        ctk.CTkLabel(self, text="Bước 1: Gán Nhãn Tự Động Với YOLO", font=ctk.CTkFont(size=20, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=15, pady=(10, 8))

        form_frame = ctk.CTkFrame(self, fg_color=COLOR_CARD)
        form_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(form_frame, text="YOLO Model (.pt):", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.entry_model = ctk.CTkEntry(form_frame, width=380, placeholder_text="Đường dẫn weights model...")
        self.entry_model.grid(row=0, column=1, padx=5, pady=8)
        ctk.CTkButton(form_frame, text="Duyệt...", width=70, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, command=self._browse_model).grid(row=0, column=2, padx=10, pady=8)

        ctk.CTkLabel(form_frame, text="Thư mục ảnh gốc:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.entry_input = ctk.CTkEntry(form_frame, width=380, placeholder_text="Thư mục chứa ảnh cần gán nhãn...")
        self.entry_input.grid(row=1, column=1, padx=5, pady=8)
        ctk.CTkButton(form_frame, text="Duyệt...", width=70, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, command=self._browse_input).grid(row=1, column=2, padx=10, pady=8)

        ctk.CTkLabel(form_frame, text="Nơi lưu done_labels:", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, padx=10, pady=8, sticky="w")
        self.entry_output = ctk.CTkEntry(form_frame, width=380, placeholder_text="Thư mục sẽ chứa kết quả done_labels...")
        self.entry_output.grid(row=2, column=1, padx=5, pady=8)
        ctk.CTkButton(form_frame, text="Duyệt...", width=70, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, command=self._browse_output).grid(row=2, column=2, padx=10, pady=8)

        p_frame = ctk.CTkFrame(self, fg_color=COLOR_CARD)
        p_frame.pack(fill="x", padx=15, pady=6)
        ctk.CTkLabel(p_frame, text="Confidence Threshold:").pack(side="left", padx=10, pady=8)
        self.slider_conf = ctk.CTkSlider(p_frame, from_=0.05, to=1.0, number_of_steps=19, progress_color=COLOR_ACCENT)
        self.slider_conf.set(0.25)
        self.slider_conf.pack(side="left", fill="x", expand=True, padx=10)
        self.lbl_conf = ctk.CTkLabel(p_frame, text="0.25", width=40, font=ctk.CTkFont(weight="bold"))
        self.lbl_conf.pack(side="right", padx=10)
        self.slider_conf.configure(command=lambda v: self.lbl_conf.configure(text=f"{v:.2f}"))

        self.progress_bar = ctk.CTkProgressBar(self, progress_color=COLOR_ACCENT)
        self.progress_bar.pack(fill="x", padx=15, pady=(10, 4))
        self.progress_bar.set(0)

        self.lbl_status = ctk.CTkLabel(self, text="Sẵn sàng thực thi", text_color="gray75")
        self.lbl_status.pack(anchor="w", padx=15, pady=2)

        self.txt_log = ctk.CTkTextbox(self, height=130, font=("Consolas", 12))
        self.txt_log.pack(fill="both", expand=True, padx=15, pady=5)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=10)

        self.btn_run = ctk.CTkButton(btn_row, text="BẮT ĐẦU GÁN NHÃN TỰ ĐỘNG", height=38, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, font=ctk.CTkFont(weight="bold"), command=self._start_thread)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_next_step = ctk.CTkButton(btn_row, text="Sang Tăng Cường Data ➔", height=38, fg_color="#2E7D32", hover_color="#1B5E20", font=ctk.CTkFont(weight="bold"), command=lambda: self.controller.show_step("augment"))
        self.btn_next_step.pack(side="right", padx=(6, 0))

    def _browse_model(self):
        f = filedialog.askopenfilename(filetypes=[("YOLO Weights", "*.pt")])
        if f:
            self.entry_model.delete(0, "end")
            self.entry_model.insert(0, f)

    def _browse_input(self):
        d = filedialog.askdirectory()
        if d:
            self.entry_input.delete(0, "end")
            self.entry_input.insert(0, d)

    def _browse_output(self):
        d = filedialog.askdirectory()
        if d:
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, d)

    def _log(self, msg):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")

    def _start_thread(self):
        if self.is_processing:
            return
        m_path = self.entry_model.get().strip()
        in_dir = self.entry_input.get().strip()
        out_dir = self.entry_output.get().strip()

        if not os.path.isfile(m_path):
            messagebox.showerror("Lỗi", "Chưa chọn file YOLO .pt hợp lệ!")
            return
        if not os.path.isdir(in_dir):
            messagebox.showerror("Lỗi", "Thư mục ảnh gốc không tồn tại!")
            return
        if not os.path.isdir(out_dir):
            messagebox.showerror("Lỗi", "Thư mục lưu kết quả không tồn tại!")
            return

        self.is_processing = True
        self.btn_run.configure(state="disabled")
        self.progress_bar.set(0)
        self.txt_log.delete("1.0", "end")

        t = threading.Thread(target=self._process_labeling, args=(m_path, in_dir, out_dir, self.slider_conf.get()), daemon=True)
        t.start()

    def _process_labeling(self, model_path, in_dir, out_dir, conf_thresh):
        try:
            from ultralytics import YOLO
            self._log(f"[+] Tải mô hình: {os.path.basename(model_path)}")
            model = YOLO(model_path)
            names_dict = model.names
            class_names = [names_dict[i] for i in sorted(names_dict.keys())]
            nc = len(class_names)
            self._log(f"[+] Classes ({nc}): {class_names}")

            base_done = os.path.join(out_dir, "done_labels")
            img_out = os.path.join(base_done, "images")
            lbl_out = os.path.join(base_done, "labels")
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)

            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            imgs = [f for f in os.listdir(in_dir) if os.path.splitext(f)[1].lower() in valid_exts]
            total = len(imgs)
            if total == 0:
                self._log("[!] Không có ảnh hợp lệ trong thư mục.")
                return

            self._log(f"[+] Bắt đầu gán nhãn cho {total} ảnh...")
            for idx, fn in enumerate(imgs):
                f_path = os.path.join(in_dir, fn)
                res = model.predict(source=f_path, conf=conf_thresh, verbose=False)[0]

                shutil.copy2(f_path, os.path.join(img_out, fn))
                base_name = os.path.splitext(fn)[0]
                lines = []
                if res.boxes is not None and len(res.boxes) > 0:
                    xywhn = res.boxes.xywhn.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy().astype(int)
                    for c_id, box in zip(classes, xywhn):
                        xc, yc, w, h = box
                        lines.append(f"{c_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                with open(os.path.join(lbl_out, f"{base_name}.txt"), "w", encoding="utf-8") as lf:
                    lf.writelines(lines)

                p = (idx + 1) / total
                self.progress_bar.set(p)
                self.lbl_status.configure(text=f"Đang xử lý: {idx + 1}/{total} ({fn})")

            # yaml
            yaml_path = os.path.join(base_done, "data.yaml")
            with open(yaml_path, "w", encoding="utf-8") as yf:
                yaml.dump({"train": "../train/images", "val": "../valid/images", "test": "../test/images", "nc": nc, "names": class_names}, yf, sort_keys=False)

            self._log(f"[✓] Gán nhãn hoàn tất! Dữ liệu tại: {base_done}")
            self.lbl_status.configure(text="Đã hoàn tất quá trình gán nhãn")

            # Chuyển sẵn thư mục sang các bước tiếp theo
            self.controller.shared_done_labels_dir = base_done
            self.after(200, lambda: self.controller.open_annotation_editor(base_done, class_names))

        except Exception as e:
            self._log(f"[x] Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", str(e))
        finally:
            self.is_processing = False
            self.btn_run.configure(state="normal")


class AugmentationView(ctk.CTkFrame):
    """View 2: Tăng cường dữ liệu"""
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.data_dir = ""
        self.class_names = []
        self.orig_cv_img = None
        self.orig_boxes = []
        self.sample_filename = ""
        self.is_processing = False

        self.applied_configs = {"crop": 0.0, "rotation": 0.0, "saturation": 0.0, "motion_blur": 0.0, "blur": 0.0}
        self.aug_metadata = [
            ("crop", "Crop (Cắt xén / Zoom)", "Cắt cúp ngẫu nhiên và tự động resize bounding box"),
            ("rotation", "Rotation (Xoay góc)", "Xoay ảnh và tính toán lại toạ độ xoay của bounding box"),
            ("saturation", "Saturation (Màu sắc)", "Điều chỉnh sắc thái màu và tạo biến thể tối / sáng"),
            ("motion_blur", "Motion Blur (Mờ chuyển động)", "Giả lập hiện tượng camera rung lắc hoặc vật chuyển động"),
            ("blur", "Gaussian Blur (Làm mờ nét)", "Làm mờ mịn các chi tiết bề mặt ảnh")
        ]
        self.status_labels = {}

        ctk.CTkLabel(self, text="Bước 2: Tăng Cường Dữ Liệu YOLO", font=ctk.CTkFont(size=20, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=15, pady=(10, 8))

        top_bar = ctk.CTkFrame(self, fg_color=COLOR_CARD)
        top_bar.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(top_bar, text="done_labels:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.entry_dir = ctk.CTkEntry(top_bar, width=360, placeholder_text="Chọn thư mục chứa done_labels...")
        self.entry_dir.grid(row=0, column=1, padx=5, pady=8)
        ctk.CTkButton(top_bar, text="Duyệt...", width=70, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, command=self._browse_dir).grid(row=0, column=2, padx=4, pady=8)
        ctk.CTkButton(top_bar, text="Đổi mẫu", width=75, fg_color="#37474F", hover_color="#263238", command=self._pick_sample).grid(row=0, column=3, padx=4, pady=8)

        list_frame = ctk.CTkScrollableFrame(self, height=270, fg_color=COLOR_CARD, label_text="DANH SÁCH BỘ TĂNG CƯỜNG (CHỌN ĐỂ TÙY BIẾN)")
        list_frame.pack(fill="x", padx=15, pady=6)

        for key, name, desc in self.aug_metadata:
            card = ctk.CTkFrame(list_frame, fg_color="#132034")
            card.pack(fill="x", padx=4, pady=4)

            info_col = ctk.CTkFrame(card, fg_color="transparent")
            info_col.pack(side="left", fill="both", expand=True, padx=8, pady=4)
            ctk.CTkLabel(info_col, text=name, font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w")
            ctk.CTkLabel(info_col, text=desc, font=ctk.CTkFont(size=10), text_color="gray75").pack(anchor="w")

            status_lbl = ctk.CTkLabel(card, text="Chưa chọn (0%)", width=130, font=ctk.CTkFont(size=11), text_color="gray60")
            status_lbl.pack(side="left", padx=6)
            self.status_labels[key] = status_lbl

            btn_cfg = ctk.CTkButton(card, text="Tùy biến %", width=85, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, command=lambda k=key, n=name: self._open_modal(k, n))
            btn_cfg.pack(side="right", padx=8, pady=6)

        bottom_frame = ctk.CTkFrame(self, fg_color=COLOR_CARD)
        bottom_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(bottom_frame, text="Nhân số lượng x").pack(side="left", padx=(10, 5), pady=8)
        self.cmb_mult = ctk.CTkComboBox(bottom_frame, values=["1", "2", "3", "5"], width=65)
        self.cmb_mult.set("2")
        self.cmb_mult.pack(side="left", padx=5)

        self.lbl_selected_summary = ctk.CTkLabel(bottom_frame, text="Đã chọn: 0 loại", text_color=COLOR_ACCENT_LIGHT, font=ctk.CTkFont(weight="bold"))
        self.lbl_selected_summary.pack(side="left", padx=10)

        self.progress_bar = ctk.CTkProgressBar(self, progress_color=COLOR_ACCENT)
        self.progress_bar.pack(fill="x", padx=15, pady=(8, 2))
        self.progress_bar.set(0)

        self.lbl_status = ctk.CTkLabel(self, text="Chọn thư mục done_labels để bắt đầu", text_color="gray70", font=ctk.CTkFont(size=11))
        self.lbl_status.pack(anchor="w", padx=15, pady=(0, 4))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=8)

        self.btn_run = ctk.CTkButton(btn_row, text="TẠO TOÀN BỘ DATASET TĂNG CƯỜNG", height=38, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, font=ctk.CTkFont(weight="bold"), command=self._start_batch_thread)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_next_step = ctk.CTkButton(btn_row, text="Sang Phân Chia Data ➔", height=38, fg_color="#2E7D32", hover_color="#1B5E20", font=ctk.CTkFont(weight="bold"), command=lambda: self.controller.show_step("split"))
        self.btn_next_step.pack(side="right", padx=(6, 0))

    def sync_shared_dir(self, directory):
        if directory and os.path.exists(directory):
            self.data_dir = directory
            self.entry_dir.delete(0, "end")
            self.entry_dir.insert(0, directory)
            self._load_yaml()
            self._pick_sample()

    def _browse_dir(self):
        p = filedialog.askdirectory(title="Chọn thư mục done_labels")
        if p:
            self.sync_shared_dir(p)

    def _load_yaml(self):
        yp = os.path.join(self.data_dir, "data.yaml")
        if os.path.exists(yp):
            with open(yp, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
                if d and "names" in d:
                    self.class_names = d["names"]

    def _pick_sample(self):
        if not self.data_dir:
            return
        img_dir = os.path.join(self.data_dir, "images")
        lbl_dir = os.path.join(self.data_dir, "labels")
        if not os.path.isdir(img_dir):
            messagebox.showwarning("Cảnh báo", "Không tìm thấy thư mục 'images'!")
            return

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        imgs = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        if not imgs:
            return

        self.sample_filename = random.choice(imgs)
        self.orig_cv_img = cv2.imread(os.path.join(img_dir, self.sample_filename))
        self.orig_boxes = []

        lp = os.path.join(lbl_dir, f"{os.path.splitext(self.sample_filename)[0]}.txt")
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) == 5:
                        self.orig_boxes.append([int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])

        self.lbl_status.configure(text=f"Mẫu xem thử: {self.sample_filename}")

    def _open_modal(self, key, name):
        if self.orig_cv_img is None:
            messagebox.showwarning("Chưa có ảnh", "Hãy chọn thư mục done_labels chứa ảnh trước!")
            return
        current_val = self.applied_configs.get(key, 0.0)
        AugmentationModal(self, key, name, self.orig_cv_img, self.orig_boxes, self.class_names, current_val, self._on_apply)

    def _on_apply(self, key, intensity):
        self.applied_configs[key] = intensity
        pct = int(intensity * 100)
        self.status_labels[key].configure(text=f"✔ Bật ({pct}%)", text_color="#00E676")
        cnt = sum(1 for v in self.applied_configs.values() if v > 0)
        self.lbl_selected_summary.configure(text=f"Đã chọn: {cnt} loại")

    def _start_batch_thread(self):
        if self.is_processing:
            return
        if not self.data_dir or not os.path.isdir(self.data_dir):
            messagebox.showerror("Lỗi", "Chưa chọn thư mục dữ liệu nguồn!")
            return

        active_augs = {k: v for k, v in self.applied_configs.items() if v > 0}
        if not active_augs:
            messagebox.showwarning("Chưa chọn", "Bạn chưa bấm 'Xác nhận áp dụng' cho bộ lọc nào!")
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

            variants_map = {"rotation": ("left", "right"), "saturation": ("dark", "bright")}
            total_per_file = 1 + sum(mult * len(variants_map.get(k, (None,))) for k in active_augs)
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

                shutil.copy2(im_path, os.path.join(out_img, f))
                if os.path.exists(lb_path):
                    shutil.copy2(lb_path, os.path.join(out_lbl, f"{bname}.txt"))
                cnt += 1
                self.progress_bar.set(cnt / total)

                for aug_key, intensity in active_augs.items():
                    variants = variants_map.get(aug_key, (None,))
                    for i in range(1, mult + 1):
                        for variant in variants:
                            aug_im, aug_bx = SingleAugmentationEngine.apply_single(aug_key, img, boxes, intensity, is_preview=False, variant=variant)
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

            src_yaml = os.path.join(self.data_dir, "data.yaml")
            if os.path.exists(src_yaml):
                shutil.copy2(src_yaml, os.path.join(out_root, "data.yaml"))

            self.lbl_status.configure(text="Đã hoàn tất toàn bộ tăng cường!")
            self.controller.shared_augmented_dir = out_root
            messagebox.showinfo("Thành công", f"Dataset đã tăng cường thành công tại:\n{out_root}")

        except Exception as e:
            messagebox.showerror("Lỗi", str(e))
        finally:
            self.is_processing = False
            self.btn_run.configure(state="normal")


class SplitDatasetView(ctk.CTkFrame):
    """View 3: Phân chia Dataset Train / Valid / Test"""
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.is_processing = False

        ctk.CTkLabel(self, text="Bước 3: Phân Chia Tập Train / Valid / Test", font=ctk.CTkFont(size=20, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=15, pady=(10, 8))

        folder_frame = ctk.CTkFrame(self, fg_color=COLOR_CARD)
        folder_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(folder_frame, text="Thư mục nguồn:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.entry_src = ctk.CTkEntry(folder_frame, width=380, placeholder_text="Chọn done_labels hoặc done_labels_augmented...")
        self.entry_src.grid(row=0, column=1, padx=5, pady=8)
        ctk.CTkButton(folder_frame, text="Duyệt...", width=70, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, command=self._browse_src).grid(row=0, column=2, padx=10, pady=8)

        ctk.CTkLabel(folder_frame, text="Nơi lưu kết quả:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.entry_dest = ctk.CTkEntry(folder_frame, width=380, placeholder_text="Thư mục chứa train_yolo_data...")
        self.entry_dest.grid(row=1, column=1, padx=5, pady=8)
        ctk.CTkButton(folder_frame, text="Duyệt...", width=70, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, command=self._browse_dest).grid(row=1, column=2, padx=10, pady=8)

        ratio_frame = ctk.CTkFrame(self, fg_color=COLOR_CARD)
        ratio_frame.pack(fill="x", padx=15, pady=6)

        ctk.CTkLabel(ratio_frame, text="Tùy chỉnh tỷ lệ (%)", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_MAIN).grid(row=0, column=0, columnspan=3, padx=10, pady=(6, 4), sticky="w")

        # Train
        ctk.CTkLabel(ratio_frame, text="Train:").grid(row=1, column=0, padx=10, pady=3, sticky="w")
        self.slider_train = ctk.CTkSlider(ratio_frame, from_=0, to=100, number_of_steps=100, progress_color=COLOR_ACCENT, command=self._update_ratio_labels)
        self.slider_train.set(80)
        self.slider_train.grid(row=1, column=1, padx=8, pady=3, sticky="ew")
        self.lbl_train = ctk.CTkLabel(ratio_frame, text="80%", width=45)
        self.lbl_train.grid(row=1, column=2, padx=5, pady=3)

        # Valid
        ctk.CTkLabel(ratio_frame, text="Valid:").grid(row=2, column=0, padx=10, pady=3, sticky="w")
        self.slider_val = ctk.CTkSlider(ratio_frame, from_=0, to=100, number_of_steps=100, progress_color=COLOR_ACCENT, command=self._update_ratio_labels)
        self.slider_val.set(15)
        self.slider_val.grid(row=2, column=1, padx=8, pady=3, sticky="ew")
        self.lbl_val = ctk.CTkLabel(ratio_frame, text="15%", width=45)
        self.lbl_val.grid(row=2, column=2, padx=5, pady=3)

        # Test
        ctk.CTkLabel(ratio_frame, text="Test:").grid(row=3, column=0, padx=10, pady=3, sticky="w")
        self.slider_test = ctk.CTkSlider(ratio_frame, from_=0, to=100, number_of_steps=100, progress_color=COLOR_ACCENT, command=self._update_ratio_labels)
        self.slider_test.set(5)
        self.slider_test.grid(row=3, column=1, padx=8, pady=3, sticky="ew")
        self.lbl_test = ctk.CTkLabel(ratio_frame, text="5%", width=45)
        self.lbl_test.grid(row=3, column=2, padx=5, pady=3)
        ratio_frame.columnconfigure(1, weight=1)

        opt_frame = ctk.CTkFrame(self, fg_color=COLOR_CARD)
        opt_frame.pack(fill="x", padx=15, pady=4)
        self.lbl_total_ratio = ctk.CTkLabel(opt_frame, text="Tổng: 100% (Hợp lệ)", font=ctk.CTkFont(weight="bold"), text_color="#00E676")
        self.lbl_total_ratio.pack(side="left", padx=15, pady=6)
        self.chk_shuffle = ctk.CTkCheckBox(opt_frame, text="Xáo trộn ngẫu nhiên (Shuffle)", fg_color=COLOR_ACCENT)
        self.chk_shuffle.select()
        self.chk_shuffle.pack(side="right", padx=15, pady=6)

        self.progress_bar = ctk.CTkProgressBar(self, progress_color=COLOR_ACCENT)
        self.progress_bar.pack(fill="x", padx=15, pady=(8, 2))
        self.progress_bar.set(0)

        self.lbl_status = ctk.CTkLabel(self, text="Sẵn sàng phân chia", text_color="gray70", font=ctk.CTkFont(size=11))
        self.lbl_status.pack(anchor="w", padx=15, pady=1)

        self.txt_log = ctk.CTkTextbox(self, height=110, font=("Consolas", 12))
        self.txt_log.pack(fill="both", expand=True, padx=15, pady=5)

        self.btn_run = ctk.CTkButton(self, text="CHIA TẬP DỮ LIỆU & TẠO DATA.YAML", height=38, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, font=ctk.CTkFont(weight="bold"), command=self._start_split_thread)
        self.btn_run.pack(fill="x", padx=15, pady=10)

    def sync_shared_dir(self, directory):
        if directory and os.path.exists(directory):
            self.entry_src.delete(0, "end")
            self.entry_src.insert(0, directory)

    def _browse_src(self):
        p = filedialog.askdirectory(title="Chọn thư mục dữ liệu nguồn")
        if p:
            self.sync_shared_dir(p)

    def _browse_dest(self):
        p = filedialog.askdirectory(title="Chọn nơi lưu train_yolo_data")
        if p:
            self.entry_dest.delete(0, "end")
            self.entry_dest.insert(0, p)

    def _update_ratio_labels(self, _=None):
        r_tr = int(self.slider_train.get())
        r_va = int(self.slider_val.get())
        r_te = int(self.slider_test.get())
        self.lbl_train.configure(text=f"{r_tr}%")
        self.lbl_val.configure(text=f"{r_va}%")
        self.lbl_test.configure(text=f"{r_te}%")
        total = r_tr + r_va + r_te
        if total == 100:
            self.lbl_total_ratio.configure(text=f"Tổng: {total}% (Hợp lệ)", text_color="#00E676")
        else:
            self.lbl_total_ratio.configure(text=f"Tổng: {total}% (Yêu cầu đúng 100%)", text_color="#FF5252")

    def _log(self, text):
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
        if not os.path.exists(os.path.join(src_dir, "images")) or not os.path.exists(os.path.join(src_dir, "labels")):
            messagebox.showerror("Lỗi", "Thư mục nguồn phải chứa đủ cả hai thư mục con 'images' và 'labels'!")
            return
        if not os.path.isdir(dest_dir):
            messagebox.showerror("Lỗi", "Thư mục lưu đích không tồn tại!")
            return

        r_tr = int(self.slider_train.get())
        r_va = int(self.slider_val.get())
        r_te = int(self.slider_test.get())
        if r_tr + r_va + r_te != 100:
            messagebox.showerror("Lỗi tỷ lệ", "Tổng 3 tập phải bằng 100%!")
            return

        self.is_processing = True
        self.btn_run.configure(state="disabled")
        self.progress_bar.set(0)
        self.txt_log.delete("1.0", "end")

        t = threading.Thread(target=self._process_split, args=(src_dir, dest_dir, r_tr, r_va, r_te, self.chk_shuffle.get() == 1), daemon=True)
        t.start()

    def _process_split(self, src_dir, dest_dir, r_train, r_val, r_test, is_shuffle):
        try:
            img_dir = os.path.join(src_dir, "images")
            lbl_dir = os.path.join(src_dir, "labels")

            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            imgs = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts]

            paired = []
            for img_f in imgs:
                lbl_f = f"{os.path.splitext(img_f)[0]}.txt"
                if os.path.exists(os.path.join(lbl_dir, lbl_f)):
                    paired.append((img_f, lbl_f))
                else:
                    self._log(f"[!] Không có nhãn cho {img_f}, bỏ qua.")

            total = len(paired)
            if total == 0:
                self._log("[x] Không tìm thấy cặp ảnh - nhãn hợp lệ nào.")
                return

            self._log(f"[+] Tìm thấy {total} mẫu dữ liệu hợp lệ.")
            if is_shuffle:
                random.seed(42)
                random.shuffle(paired)

            n_train = int(total * (r_train / 100.0))
            n_val = int(total * (r_val / 100.0))
            splits = {
                'train': paired[:n_train],
                'valid': paired[n_train:n_train + n_val],
                'test': paired[n_train + n_val:]
            }

            target_root = os.path.join(dest_dir, "train_yolo_data")
            for sp in ['train', 'valid', 'test']:
                os.makedirs(os.path.join(target_root, sp, "images"), exist_ok=True)
                os.makedirs(os.path.join(target_root, sp, "labels"), exist_ok=True)

            copied = 0
            for sp_name, files in splits.items():
                for im_f, lb_f in files:
                    shutil.copy2(os.path.join(img_dir, im_f), os.path.join(target_root, sp_name, "images", im_f))
                    shutil.copy2(os.path.join(lbl_dir, lb_f), os.path.join(target_root, sp_name, "labels", lb_f))
                    copied += 1
                    self.progress_bar.set(copied / total)
                    self.lbl_status.configure(text=f"Đang sao chép ({sp_name}): {copied}/{total}")

            # Ghi yaml tương đối chuẩn YOLO
            src_yaml = os.path.join(src_dir, "data.yaml")
            nc, names = 0, []
            if os.path.exists(src_yaml):
                with open(src_yaml, "r", encoding="utf-8") as yf:
                    yd = yaml.safe_load(yf)
                    if yd:
                        nc = yd.get("nc", 0)
                        names = yd.get("names", [])

            out_yaml = os.path.join(target_root, "data.yaml")
            with open(out_yaml, "w", encoding="utf-8") as yf:
                yaml.dump({
                    "train": "./train/images",
                    "val": "./valid/images",
                    "test": "./test/images",
                    "nc": nc if nc > 0 else len(names),
                    "names": names
                }, yf, sort_keys=False)

            self._log(f"[✓] Đã tạo data.yaml tại: {out_yaml}")
            self._log(f"[✓] Hoàn tất chia dataset tại: {target_root}")
            messagebox.showinfo("Thành công", f"Chia dữ liệu thành công!\n{target_root}")

        except Exception as e:
            self._log(f"[x] Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", str(e))
        finally:
            self.is_processing = False
            self.btn_run.configure(state="normal")
            self.lbl_status.configure(text="Trạng thái: Hoàn tất")


# =================================================================
# 4. APP CHÍNH TÍCH HỢP 3 CỘT (MAIN CONTAINER)
# =================================================================
class IntegratedYOLOStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hệ Thống Tiền Xử Lý & Gán Nhãn Tự Động YOLO")
        self.geometry("1280x760")
        self.minsize(1180, 700)

        # Lưu vết đường dẫn trung gian giữa các bước
        self.shared_done_labels_dir = ""
        self.shared_augmented_dir = ""

        self._create_layout()
        self.show_step("label")

    def _create_avatar_image(self):
        avatar_path = "tacgia.png"  
        
        if os.path.exists(avatar_path):
            img = Image.open(avatar_path)
            return ctk.CTkImage(light_image=img, dark_image=img, size=(200, 270))
        else:
            # Dự phòng: nếu chưa có file ảnh thì tự vẽ avatar mặc định
            size = (110, 110)
            img = Image.new("RGBA", size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.ellipse([5, 5, 105, 105], fill="#1E88E5", outline="#90CAF9", width=3)
            draw.ellipse([37, 20, 73, 56], fill="#E3F2FD")
            draw.chord([20, 52, 90, 115], start=0, end=180, fill="#E3F2FD")
            return ctk.CTkImage(light_image=img, dark_image=img, size=(100, 100))
    def _load_system_banner(self):
        banner_path = "LACHONG.png"  # <-- ĐIỀN ĐƯỜNG DẪN ẢNH CỦA BẠN (.png, .jpg)
        
        if os.path.exists(banner_path):
            img = Image.open(banner_path)
            # Tự động co giãn theo chiều ngang cột phải (~220px) mà vẫn giữ tỉ lệ ảnh
            orig_w, orig_h = img.size
            target_w = 210
            target_h = int(orig_h * (target_w / orig_w))
            return ctk.CTkImage(light_image=img, dark_image=img, size=(target_w, target_h))
        else:
            # Tạo ảnh placeholder màu xanh biển nếu chưa có file ảnh
            img = Image.new("RGBA", (210, 120), "#1C2D4A")
            return ctk.CTkImage(light_image=img, dark_image=img, size=(210, 120))

    def _create_layout(self):
        # Thiết lập Grid 3 Cột: Trái (Menu), Giữa (Workspace), Phải (Info & Author)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=210)  # Cột trái
        self.grid_columnconfigure(1, weight=1)              # Cột giữa
        self.grid_columnconfigure(2, weight=0, minsize=250)  # Cột phải

        # -------------------------------------------------------------
        # CỘT TRÁI: ĐIỀU HƯỚNG CÁC TRANG QUY TRÌNH
        # -------------------------------------------------------------
        self.left_panel = ctk.CTkFrame(self, fg_color=COLOR_BG_SIDEBAR, corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 2), pady=0)

        ctk.CTkLabel(self.left_panel, text="QUY TRÌNH", font=ctk.CTkFont(size=15, weight="bold"), text_color=COLOR_ACCENT_LIGHT).pack(anchor="w", padx=20, pady=(25, 15))

        self.nav_btns = {}
        steps = [
            ("label", "1. Gán nhãn tự động"),
            ("augment", "2. Tăng cường data"),
            ("split", "3. Phân chia tập data")
        ]

        for key, title in steps:
            btn = ctk.CTkButton(
                self.left_panel,
                text=title,
                height=42,
                anchor="w",
                fg_color="transparent",
                text_color=COLOR_TEXT_MAIN,
                hover_color="#1E304D",
                font=ctk.CTkFont(size=13, weight="bold"),
                command=lambda k=key: self.show_step(k)
            )
            btn.pack(fill="x", padx=10, pady=5)
            self.nav_btns[key] = btn

        # Info nhỏ hỗ trợ luồng dữ liệu
        pipeline_box = ctk.CTkFrame(self.left_panel, fg_color="#132034")
        pipeline_box.pack(fill="x", padx=12, pady=(35, 10))
        ctk.CTkLabel(pipeline_box, text="DÒNG CHẢY DỮ LIỆU", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_ACCENT_LIGHT).pack(anchor="w", padx=10, pady=(8, 4))
        desc = "• Gán nhãn -> done_labels\n• Tăng cường -> augmented\n• Phân chia -> train_yolo_data"
        ctk.CTkLabel(pipeline_box, text=desc, justify="left", font=ctk.CTkFont(size=10), text_color="gray75").pack(anchor="w", padx=10, pady=(0, 8))

        # -------------------------------------------------------------
        # CỘT GIỮA: VÙNG XỬ LÝ LOGIC CHÍNH
        # -------------------------------------------------------------
        self.center_panel = ctk.CTkFrame(self, fg_color=COLOR_BG_MAIN, corner_radius=0)
        self.center_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.center_panel.grid_rowconfigure(0, weight=1)
        self.center_panel.grid_columnconfigure(0, weight=1)

        # Đăng ký 3 View tương ứng
        self.views = {
            "label": AutoLabelView(self.center_panel, self),
            "augment": AugmentationView(self.center_panel, self),
            "split": SplitDatasetView(self.center_panel, self)
        }
        for view in self.views.values():
            view.grid(row=0, column=0, sticky="nsew")

        # -------------------------------------------------------------
        # CỘT PHẢI: HEADER "TOOL GẮN NHÃN TỰ ĐỘNG" & TÁC GIẢ Ở DƯỚI
        # -------------------------------------------------------------
        self.right_panel = ctk.CTkFrame(self, fg_color=COLOR_BG_SIDEBAR, corner_radius=0)
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(2, 0), pady=0)

        # Header trên cùng
        title_box = ctk.CTkFrame(self.right_panel, fg_color=COLOR_CARD)
        title_box.pack(fill="x", padx=12, pady=(20, 10))
        ctk.CTkLabel(
            title_box, 
            text="Tool gắn nhãn tự động", 
            font=ctk.CTkFont(size=16, weight="bold"), 
            text_color=COLOR_ACCENT_LIGHT, 
            wraplength=210
        ).pack(padx=10, pady=12)

        # Thống kê nhanh / Quick info
        # Khung hiển thị ảnh thay thế cho Môi trường & Hệ thống
        banner_box = ctk.CTkFrame(self.right_panel, fg_color=COLOR_CARD, corner_radius=8)
        banner_box.pack(fill="x", padx=12, pady=5)

        self.system_banner_img = self._load_system_banner()
        self.lbl_system_banner = ctk.CTkLabel(banner_box, text="", image=self.system_banner_img)
        self.lbl_system_banner.pack(padx=8, pady=8)

        # Khoảng đệm để đẩy khối tác giả xuống đáy
        spacer = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        # Khối hình người và thông tin tác giả ở dưới cùng
        author_frame = ctk.CTkFrame(self.right_panel, fg_color=COLOR_CARD, corner_radius=10)
        author_frame.pack(fill="x", padx=12, pady=(10, 20))

        # Avatar người
        self.author_img = self._create_avatar_image()
        lbl_avatar = ctk.CTkLabel(author_frame, text="", image=self.author_img)
        lbl_avatar.pack(pady=(12, 6))

        ctk.CTkLabel(author_frame, text="Tác giả: Hàn Quốc Bảo", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(pady=(2, 1))
        ctk.CTkLabel(author_frame, text="Lạc Hồng", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_ACCENT_LIGHT).pack(pady=(0, 12))

    def show_step(self, step_key):
        """Chuyển đổi view giữa 3 module và tự động đồng bộ hóa đường dẫn."""
        for key, btn in self.nav_btns.items():
            if key == step_key:
                btn.configure(fg_color=COLOR_ACCENT, text_color="#FFFFFF")
            else:
                btn.configure(fg_color="transparent", text_color=COLOR_TEXT_MAIN)

        self.views[step_key].tkraise()

        # Tự động truyền dữ liệu thư mục sang View kế tiếp
        if step_key == "augment" and self.shared_done_labels_dir:
            self.views["augment"].sync_shared_dir(self.shared_done_labels_dir)
        elif step_key == "split":
            # Ưu tiên lấy thư mục sau tăng cường nếu có, ngược lại lấy done_labels
            target_sync = self.shared_augmented_dir or self.shared_done_labels_dir
            if target_sync:
                self.views["split"].sync_shared_dir(target_sync)

    def open_annotation_editor(self, done_dir, class_names):
        """Mở trình kiểm tra và chỉnh sửa box trực quan."""
        editor = AnnotationEditor(self, done_dir, class_names)
        editor.grab_set()


if __name__ == "__main__":
    app = IntegratedYOLOStudioApp()
    app.mainloop()