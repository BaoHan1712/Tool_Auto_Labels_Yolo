import os
import shutil
import threading
import yaml
import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
from PIL import Image, ImageTk
from ultralytics import YOLO
import sys

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
    

def resource_path(relative_path):
    """Lấy đường dẫn tuyệt đối đến tài nguyên, hỗ trợ cả khi chạy dev và qua PyInstaller."""
    try:
        # PyInstaller tạo ra một thư mục tạm và lưu đường dẫn tại _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class AnnotationEditor(ctk.CTkToplevel):
    """Cửa sổ chỉnh sửa trực quan các nhãn sau khi detect."""
    def __init__(self, parent, done_dir, class_names):
        super().__init__(parent)
        self.title("YOLO Annotation Editor & Reviewer")
        self.geometry("1280x820")
        
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
        self.display_img = None
        self.tk_img = None
        self.img_w = 0
        self.img_h = 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.boxes = []
        self.selected_box_idx = None
        
        self.drag_mode = None
        self.active_handle = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.HANDLE_SIZE = 8
        
        self.colors = ["#00FF66", "#FF3366", "#3399FF", "#FFCC00", "#FF66FF", "#00FFFF"]
        
        self._setup_ui()
        self._load_current_image()

    def _setup_ui(self):
        top_bar = ctk.CTkFrame(self)
        top_bar.pack(fill="x", padx=10, pady=5)
        
        self.btn_prev = ctk.CTkButton(top_bar, text="◀ Trước", width=80, command=self._prev_image)
        self.btn_prev.pack(side="left", padx=(5, 2), pady=5)
        
        self.lbl_counter = ctk.CTkLabel(top_bar, text="0 / 0", width=70, font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_counter.pack(side="left", padx=5)
        
        self.btn_next = ctk.CTkButton(top_bar, text="Sau ▶", width=80, command=self._next_image)
        self.btn_next.pack(side="left", padx=(2, 10), pady=5)
        
        ctk.CTkLabel(top_bar, text="Class:").pack(side="left", padx=(5, 5))
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
        
        self.canvas = Canvas(self.canvas_frame, bg="#1E1E1E", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.bind("<Delete>", lambda e: self._delete_selected_box())
        self.bind("<BackSpace>", lambda e: self._delete_selected_box())
        self.bind("<Left>", lambda e: self._prev_image())
        self.bind("<Right>", lambda e: self._next_image())
        self.bind("<Return>", lambda e: self._update_current_annotation())

    def _load_current_image(self):
        if not self.image_files:
            return
            
        file_name = self.image_files[self.current_idx]
        total_files = len(self.image_files)
        
        self.lbl_counter.configure(text=f"{self.current_idx + 1} / {total_files}")
        self.title(f"YOLO Editor & Reviewer - [{self.current_idx + 1}/{total_files}] {file_name}")
        
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
            canv_w, canv_h = 1000, 650

        ratio = min(canv_w / self.img_w, canv_h / self.img_h)
        new_w = int(self.img_w * ratio)
        new_h = int(self.img_h * ratio)
        
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

            self.canvas.create_rectangle(
                x1, y1, x2, y2, 
                outline=color, 
                width=outline_w, 
                tags="box_element"
            )

            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"ID_{cls_id}"
            tag_text = f" {cls_name} "
            self.canvas.create_rectangle(
                x1, y1 - 18, x1 + len(tag_text) * 8, y1, 
                fill=color, 
                outline=color, 
                tags="box_element"
            )
            self.canvas.create_text(
                x1 + 2, y1 - 9, 
                text=tag_text, 
                fill="#000000", 
                font=("Arial", 9, "bold"), 
                anchor="w", 
                tags="box_element"
            )

            if is_selected:
                hs = self.HANDLE_SIZE
                corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                for cx, cy in corners:
                    self.canvas.create_rectangle(
                        cx - hs, cy - hs, cx + hs, cy + hs, 
                        fill="#FFFFFF", 
                        outline=color, 
                        tags="box_element"
                    )

    def _get_hit_handle(self, box, x, y):
        hs = self.HANDLE_SIZE + 3
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
        self.drag_start_x = x
        self.drag_start_y = y

        if self.selected_box_idx is not None:
            active_box = self.boxes[self.selected_box_idx]
            handle = self._get_hit_handle(active_box, x, y)
            if handle:
                self.drag_mode = 'resize'
                self.active_handle = handle
                return

        clicked_box_idx = None
        for idx in reversed(range(len(self.boxes))):
            b = self.boxes[idx]
            min_x, max_x = min(b['x1'], b['x2']), max(b['x1'], b['x2'])
            min_y, max_y = min(b['y1'], b['y2']), max(b['y1'], b['y2'])
            if min_x <= x <= max_x and min_y <= y <= max_y:
                clicked_box_idx = idx
                break

        if clicked_box_idx is not None:
            self.selected_box_idx = clicked_box_idx
            self.drag_mode = 'move'
            self._draw_boxes()
            return

        self.selected_box_idx = None
        current_cls_name = self.cmb_class.get()
        cls_id = self.class_names.index(current_cls_name) if current_cls_name in self.class_names else 0
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
            self.drag_start_x = event.x
            self.drag_start_y = event.y

        elif self.drag_mode == 'resize':
            if 'left' in self.active_handle:
                b['x1'] += dx
            if 'right' in self.active_handle:
                b['x2'] += dx
            if 'top' in self.active_handle:
                b['y1'] += dy
            if 'bottom' in self.active_handle:
                b['y2'] += dy
            self.drag_start_x = event.x
            self.drag_start_y = event.y

        elif self.drag_mode == 'create':
            b['x2'] = event.x
            b['y2'] = event.y

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

    def _update_current_annotation(self, show_message=True):
        if not self.image_files:
            return

        self._sync_back_to_raw()
        file_name = self.image_files[self.current_idx]
        self.pending_annotations[file_name] = [
            {'cls': item['cls'], 'norm_coords': item['norm_coords'][:]}
            for item in self.raw_boxes
        ]

        if show_message:
            base_name = os.path.splitext(file_name)[0]
            messagebox.showinfo("Đã cập nhật", f"Đã cập nhật nhãn cho ảnh:\n{base_name}", parent=self)

    def _write_annotation(self, file_name, raw_boxes):
        base_name = os.path.splitext(file_name)[0]
        label_path = os.path.join(self.labels_dir, f"{base_name}.txt")

        lines = []
        for item in raw_boxes:
            cls_id = item['cls']
            x1, y1, x2, y2 = item['norm_coords']
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            xc = x1 + w / 2.0
            yc = y1 + h / 2.0
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        with open(label_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _save_all_annotations(self):
        if not self.image_files:
            return

        self._update_current_annotation(show_message=False)
        for file_name, raw_boxes in self.pending_annotations.items():
            self._write_annotation(file_name, raw_boxes)

        messagebox.showinfo(
            "Đã lưu",
            f"Đã lưu tất cả nhãn đã kiểm tra ({len(self.pending_annotations)} ảnh).",
            parent=self
        )

    def _prev_image(self):
        if self.current_idx > 0:
            self._update_current_annotation(show_message=False)
            self.current_idx -= 1
            self._load_current_image()

    def _next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self._update_current_annotation(show_message=False)
            self.current_idx += 1
            self._load_current_image()


class YOLOLabelingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("YOLO Auto-Labeling Studio")
        self.geometry("1080x720")
        self.resizable(True, True)

        self.is_processing = False
        self._setup_ui()

    def _load_image(self, file_name, size):
        """Hàm phụ trợ load ảnh CTkImage an toàn."""
        # Chuyển đổi tên file sang đường dẫn thực tế bằng resource_path
        full_path = resource_path(file_name)
        
        # Nếu không thấy trong thư mục đóng gói thì tìm thử ở thư mục hiện tại
        if not os.path.exists(full_path):
            full_path = file_name

        if os.path.exists(full_path):
            try:
                pil_img = Image.open(full_path)
                return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=size)
            except Exception as e:
                print(f"Lỗi load ảnh {full_path}: {e}")
        return None

    def _setup_ui(self):
        # Header Tiêu đề
        title_label = ctk.CTkLabel(
            self, 
            text="HỆ THỐNG GÁN NHÃN DỮ LIỆU TỰ ĐỘNG - YOLO AUTO-LABELING", 
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title_label.pack(padx=20, pady=(15, 10))

        # Main Container chia 2 cột: Trái (Thao tác/Log), Phải (Tác giả)
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=15, pady=5)

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=0)
        container.grid_rowconfigure(0, weight=1)

        # ---------------- CỘT TRÁI ----------------
        left_frame = ctk.CTkFrame(container)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)

        # Frame chọn đường dẫn
        file_frame = ctk.CTkFrame(left_frame)
        file_frame.pack(padx=15, pady=10, fill="x")
        file_frame.grid_columnconfigure(1, weight=1)

        # 1. Model Selection
        ctk.CTkLabel(file_frame, text="YOLO Model (.pt):", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.entry_model = ctk.CTkEntry(file_frame, placeholder_text="Đường dẫn file checkpoint .pt...")
        self.entry_model.grid(row=0, column=1, padx=5, pady=8, sticky="ew")
        ctk.CTkButton(file_frame, text="Duyệt", width=75, command=self._browse_model).grid(row=0, column=2, padx=10, pady=8)

        # 2. Input Images Directory
        ctk.CTkLabel(file_frame, text="Thư mục ảnh gốc:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.entry_input = ctk.CTkEntry(file_frame, placeholder_text="Thư mục chứa toàn bộ ảnh thô...")
        self.entry_input.grid(row=1, column=1, padx=5, pady=8, sticky="ew")
        ctk.CTkButton(file_frame, text="Duyệt", width=75, command=self._browse_input).grid(row=1, column=2, padx=10, pady=8)

        # 3. Output Destination
        ctk.CTkLabel(file_frame, text="Thư mục lưu kết quả:", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, padx=10, pady=8, sticky="w")
        self.entry_output = ctk.CTkEntry(file_frame, placeholder_text="Thư mục sẽ chứa folder done_labels...")
        self.entry_output.grid(row=2, column=1, padx=5, pady=8, sticky="ew")
        ctk.CTkButton(file_frame, text="Duyệt", width=75, command=self._browse_output).grid(row=2, column=2, padx=10, pady=8)

        # Thanh cấu hình ngưỡng
        param_frame = ctk.CTkFrame(left_frame)
        param_frame.pack(padx=15, pady=5, fill="x")

        ctk.CTkLabel(param_frame, text="Confidence Threshold:").grid(row=0, column=0, padx=15, pady=8, sticky="w")
        self.slider_conf = ctk.CTkSlider(param_frame, from_=0.05, to=1.0, number_of_steps=19, width=280)
        self.slider_conf.set(0.25)
        self.slider_conf.grid(row=0, column=1, padx=10, pady=8)
        self.lbl_conf_val = ctk.CTkLabel(param_frame, text="0.25", font=ctk.CTkFont(weight="bold"))
        self.lbl_conf_val.grid(row=0, column=2, padx=10, pady=8)
        self.slider_conf.configure(command=lambda val: self.lbl_conf_val.configure(text=f"{val:.2f}"))

        self.progress_bar = ctk.CTkProgressBar(left_frame)
        self.progress_bar.pack(padx=15, pady=(10, 4), fill="x")
        self.progress_bar.set(0)

        self.lbl_status = ctk.CTkLabel(left_frame, text="Trạng thái: Sẵn sàng", text_color="gray70")
        self.lbl_status.pack(padx=15, pady=(0, 4), anchor="w")

        self.txt_log = ctk.CTkTextbox(left_frame, height=130, font=("Consolas", 12))
        self.txt_log.pack(padx=15, pady=5, fill="both", expand=True)

        self.btn_run = ctk.CTkButton(
            left_frame, 
            text="Bắt đầu gán nhãn tự động", 
            font=ctk.CTkFont(size=15, weight="bold"),
            height=42,
            command=self._start_processing_thread
        )
        self.btn_run.pack(padx=15, pady=12, fill="x")

        # ---------------- CỘT PHẢI ----------------
        right_frame = ctk.CTkFrame(container, width=280)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
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
        lbl_author_title.pack(pady=(10, 15))

        # Ảnh tác giả
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

        # Đơn vị / Trường học
        lbl_school = ctk.CTkLabel(
            author_frame, 
            text="Đại học Lạc Hồng", 
            font=ctk.CTkFont(size=15, weight="bold"), 
            text_color="#FFB300"
        )
        lbl_school.pack(pady=(0, 10))

    def _browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Weights", "*.pt")])
        if path:
            self.entry_model.delete(0, "end")
            self.entry_model.insert(0, path)

    def _browse_input(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_input.delete(0, "end")
            self.entry_input.insert(0, path)

    def _browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, path)

    def _log(self, text: str):
        self.txt_log.insert("end", text + "\n")
        self.txt_log.see("end")

    def _start_processing_thread(self):
        if self.is_processing:
            return

        model_path = self.entry_model.get().strip()
        in_dir = self.entry_input.get().strip()
        out_dir = self.entry_output.get().strip()

        if not os.path.isfile(model_path):
            messagebox.showerror("Lỗi", "File model .pt không tồn tại!")
            return
        if not os.path.isdir(in_dir):
            messagebox.showerror("Lỗi", "Thư mục ảnh gốc không hợp lệ!")
            return
        if not os.path.isdir(out_dir):
            messagebox.showerror("Lỗi", "Thư mục đích không hợp lệ!")
            return

        self.is_processing = True
        self.btn_run.configure(state="disabled")
        self.progress_bar.set(0)
        self.txt_log.delete("1.0", "end")

        thread = threading.Thread(
            target=self._process_labeling,
            args=(model_path, in_dir, out_dir, self.slider_conf.get()),
            daemon=True
        )
        thread.start()

    def _process_labeling(self, model_path: str, in_dir: str, out_dir: str, conf_thresh: float):
        try:
            self._log(f"[+] Khởi tạo mô hình: {os.path.basename(model_path)}")
            model = YOLO(model_path)

            names_dict = model.names
            class_names = [names_dict[i] for i in sorted(names_dict.keys())]
            nc = len(class_names)
            self._log(f"[+] Nhận diện được {nc} classes: {class_names}")

            base_done_dir = os.path.join(out_dir, "done_labels")
            images_dir = os.path.join(base_done_dir, "images")
            labels_dir = os.path.join(base_done_dir, "labels")

            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            image_files = [
                f for f in os.listdir(in_dir) 
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]

            total = len(image_files)
            if total == 0:
                self._log("[!] Không tìm thấy ảnh hợp lệ trong thư mục nguồn.")
                self._finish_task()
                return

            self._log(f"[+] Bắt đầu auto-labeling cho {total} ảnh...")

            for idx, file_name in enumerate(image_files):
                img_path = os.path.join(in_dir, file_name)
                results = model.predict(source=img_path, conf=conf_thresh, verbose=False)
                result = results[0]

                dest_img_path = os.path.join(images_dir, file_name)
                shutil.copy2(img_path, dest_img_path)

                base_name = os.path.splitext(file_name)[0]
                label_path = os.path.join(labels_dir, f"{base_name}.txt")

                lines = []
                if result.boxes is not None and len(result.boxes) > 0:
                    xywhn = result.boxes.xywhn.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)

                    for cls_id, box in zip(classes, xywhn):
                        xc, yc, w, h = box
                        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                with open(label_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                progress = (idx + 1) / total
                self.progress_bar.set(progress)
                self.lbl_status.configure(text=f"Đang xử lý: {idx + 1}/{total} ({file_name})")

            yaml_path = os.path.join(base_done_dir, "data.yaml")
            yaml_data = {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": nc,
                "names": class_names
            }

            with open(yaml_path, "w", encoding="utf-8") as yf:
                yaml.dump(yaml_data, yf, default_flow_style=None, sort_keys=False)

            self._log(f"[✓] Đã tạo file: {yaml_path}")
            self._log(f"[✓] Auto-label hoàn tất {total} ảnh.")

            self.after(200, lambda: self._open_editor(base_done_dir, class_names))

        except Exception as e:
            self._log(f"[x] Xảy ra lỗi: {str(e)}")
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {str(e)}")
        finally:
            self._finish_task()

    def _open_editor(self, done_dir, class_names):
        editor = AnnotationEditor(self, done_dir, class_names)
        editor.grab_set()

    def _finish_task(self):
        self.is_processing = False
        self.btn_run.configure(state="normal")
        self.lbl_status.configure(text="Trạng thái: Hoàn tất quá trình gán nhãn")


if __name__ == "__main__":
    app = YOLOLabelingApp()
    app.mainloop()