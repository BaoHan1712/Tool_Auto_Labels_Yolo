import os
import shutil
import threading
import yaml
import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
from PIL import Image, ImageTk
from ultralytics import YOLO

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class AnnotationEditor(ctk.CTkToplevel):
    """Cửa sổ chỉnh sửa trực quan các nhãn sau khi detect."""
    def __init__(self, parent, done_dir, class_names):
        super().__init__(parent)
        self.title("YOLO Annotation Editor & Reviewer")
        self.geometry("1150x780")
        
        self.done_dir = done_dir
        self.images_dir = os.path.join(done_dir, "images")
        self.labels_dir = os.path.join(done_dir, "labels")
        self.class_names = class_names
        
        # Lấy danh sách ảnh
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        
        self.current_idx = 0
        self.pending_annotations = {}
        
        # Dữ liệu ảnh hiện tại
        self.orig_img = None
        self.display_img = None
        self.tk_img = None
        self.img_w = 0
        self.img_h = 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Danh sách box: mỗi item là dict: {'cls': int, 'x1': float, 'y1': float, 'x2': float, 'y2': float} (tọa độ hiển thị trên canvas)
        self.boxes = []
        self.selected_box_idx = None
        
        # Trạng thái chuột
        self.drag_mode = None  # 'move', 'resize', 'create'
        self.active_handle = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.HANDLE_SIZE = 8
        
        # Bảng màu cho từng class
        self.colors = ["#00FF66", "#FF3366", "#3399FF", "#FFCC00", "#FF66FF", "#00FFFF"]
        
        self._setup_ui()
        self._load_current_image()

    def _setup_ui(self):
        # Thanh điều hướng phía trên
        top_bar = ctk.CTkFrame(self)
        top_bar.pack(fill="x", padx=10, pady=5)
        
        self.btn_prev = ctk.CTkButton(top_bar, text="◀ Ảnh trước", width=110, command=self._prev_image)
        self.btn_prev.pack(side="left", padx=5, pady=5)
        
        self.lbl_counter = ctk.CTkLabel(top_bar, text="0 / 0", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_counter.pack(side="left", padx=15)
        
        self.btn_next = ctk.CTkButton(top_bar, text="Ảnh sau ▶", width=110, command=self._next_image)
        self.btn_next.pack(side="left", padx=5, pady=5)
        
        # Controls gán class & thao tác box
        ctk.CTkLabel(top_bar, text="Class vẽ mới:").pack(side="left", padx=(25, 5))
        self.cmb_class = ctk.CTkComboBox(top_bar, values=self.class_names, width=140)
        if self.class_names:
            self.cmb_class.set(self.class_names[0])
        self.cmb_class.pack(side="left", padx=5)
        
        self.btn_del = ctk.CTkButton(top_bar, text="Xóa Box (Del)", fg_color="#D32F2F", hover_color="#9A0007", width=110, command=self._delete_selected_box)
        self.btn_del.pack(side="left", padx=15)
        
        self.btn_save = ctk.CTkButton(top_bar, text="💾 Xác nhận & Lưu tất cả", fg_color="#2E7D32", hover_color="#1B5E20", width=160, command=self._save_all_annotations)
        self.btn_save.pack(side="right", padx=10)

        # Canvas vẽ ảnh và box
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.canvas = Canvas(self.canvas_frame, bg="#1E1E1E", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Binds chuột và bàn phím
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
        self.lbl_counter.configure(text=f"[{self.current_idx + 1}/{len(self.image_files)}] - {file_name}")
        
        img_path = os.path.join(self.images_dir, file_name)
        self.orig_img = Image.open(img_path)
        self.img_w, self.img_h = self.orig_img.size
        
        # Đọc nhãn tương ứng từ file .txt
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
                        # Đổi từ normalized xywh sang normalized x1y1x2y2
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

        # Tính scale giữ tỉ lệ khung hình (Aspect Ratio)
        ratio = min(canv_w / self.img_w, canv_h / self.img_h)
        new_w = int(self.img_w * ratio)
        new_h = int(self.img_h * ratio)
        
        self.scale_x = new_w
        self.scale_y = new_h
        self.offset_x = (canv_w - new_w) // 2
        self.offset_y = (canv_h - new_h) // 2

        # Vẽ background ảnh
        resized = self.orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_img)

        # Chuyển đổi normalized coords sang pixel coords trên canvas
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

            # Vẽ bounding box
            self.canvas.create_rectangle(
                x1, y1, x2, y2, 
                outline=color, 
                width=outline_w, 
                tags="box_element"
            )

            # Vẽ label tag
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

            # Vẽ handles ở 4 góc nếu đang được chọn để kéo resize
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
        """Kiểm tra chuột có đang trỏ vào handle để resize không."""
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

        # Kiểm tra click vào handle của box đang chọn
        if self.selected_box_idx is not None:
            active_box = self.boxes[self.selected_box_idx]
            handle = self._get_hit_handle(active_box, x, y)
            if handle:
                self.drag_mode = 'resize'
                self.active_handle = handle
                return

        # Kiểm tra click chọn box bất kỳ
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

        # Click ra vùng trống -> Tạo nhãn mới
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
            # Chuẩn hóa thứ tự toạ độ x1 < x2, y1 < y2
            x1, x2 = min(b['x1'], b['x2']), max(b['x1'], b['x2'])
            y1, y2 = min(b['y1'], b['y2']), max(b['y1'], b['y2'])
            
            # Lọc bỏ nếu box quá nhỏ (do vô tình click)
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
        """Chuyển đổi toạ độ canvas sang normalized YOLO coords."""
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
        self.geometry("740x650")
        self.resizable(False, False)

        self.is_processing = False
        self._setup_ui()

    def _setup_ui(self):
        title_label = ctk.CTkLabel(
            self, 
            text="YOLO Auto-Labeling Studio", 
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title_label.pack(padx=20, pady=(20, 10))

        file_frame = ctk.CTkFrame(self)
        file_frame.pack(padx=20, pady=10, fill="x")

        # 1. Model Selection
        ctk.CTkLabel(file_frame, text="YOLO Model (.pt):", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.entry_model = ctk.CTkEntry(file_frame, width=440, placeholder_text="Đường dẫn file checkpoint .pt...")
        self.entry_model.grid(row=0, column=1, padx=5, pady=8)
        ctk.CTkButton(file_frame, text="Duyệt", width=80, command=self._browse_model).grid(row=0, column=2, padx=10, pady=8)

        # 2. Input Images Directory
        ctk.CTkLabel(file_frame, text="Thư mục ảnh gốc:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.entry_input = ctk.CTkEntry(file_frame, width=440, placeholder_text="Thư mục chứa toàn bộ ảnh thô...")
        self.entry_input.grid(row=1, column=1, padx=5, pady=8)
        ctk.CTkButton(file_frame, text="Duyệt", width=80, command=self._browse_input).grid(row=1, column=2, padx=10, pady=8)

        # 3. Output Destination
        ctk.CTkLabel(file_frame, text="Thư mục lưu kết quả:", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, padx=10, pady=8, sticky="w")
        self.entry_output = ctk.CTkEntry(file_frame, width=440, placeholder_text="Thư mục sẽ chứa folder done_labels...")
        self.entry_output.grid(row=2, column=1, padx=5, pady=8)
        ctk.CTkButton(file_frame, text="Duyệt", width=80, command=self._browse_output).grid(row=2, column=2, padx=10, pady=8)

        param_frame = ctk.CTkFrame(self)
        param_frame.pack(padx=20, pady=10, fill="x")

        ctk.CTkLabel(param_frame, text="Confidence Threshold:").grid(row=0, column=0, padx=15, pady=10, sticky="w")
        self.slider_conf = ctk.CTkSlider(param_frame, from_=0.05, to=1.0, number_of_steps=19, width=300)
        self.slider_conf.set(0.25)
        self.slider_conf.grid(row=0, column=1, padx=10, pady=10)
        self.lbl_conf_val = ctk.CTkLabel(param_frame, text="0.25")
        self.lbl_conf_val.grid(row=0, column=2, padx=10, pady=10)
        self.slider_conf.configure(command=lambda val: self.lbl_conf_val.configure(text=f"{val:.2f}"))

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.pack(padx=20, pady=(15, 5), fill="x")
        self.progress_bar.set(0)

        self.lbl_status = ctk.CTkLabel(self, text="Trạng thái: Sẵn sàng", text_color="gray70")
        self.lbl_status.pack(padx=20, pady=(0, 5), anchor="w")

        self.txt_log = ctk.CTkTextbox(self, height=140, font=("Consolas", 12))
        self.txt_log.pack(padx=20, pady=5, fill="both", expand=True)

        self.btn_run = ctk.CTkButton(
            self, 
            text="Bắt đầu gán nhãn tự động", 
            font=ctk.CTkFont(size=15, weight="bold"),
            height=40,
            command=self._start_processing_thread
        )
        self.btn_run.pack(padx=20, pady=15, fill="x")

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

                # Copy ảnh vào done_labels/images
                dest_img_path = os.path.join(images_dir, file_name)
                shutil.copy2(img_path, dest_img_path)

                # Lưu nhãn vào done_labels/labels
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

            # Tạo file data.yaml
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

            # Mở cửa sổ chỉnh sửa trên Main Thread
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