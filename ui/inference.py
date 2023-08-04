import os
import subprocess

import tkinter as tk
import customtkinter as ctk
from tkinter import ttk
from tkinter.filedialog import askopenfile, askdirectory

from ui.buttons import MenuButton
from ui.labels import HeaderLabel
from processing.utils import load_cache, save_cache, VideoScanner
from processing.yolo_api import YOLO_Call

INFERENCE_CACHE_NAME = "inference.cache"
class InferenceFrame(ctk.CTkFrame):
    def __init__(self, root, back_cmd=lambda: None, open_progress_page=lambda: None):
        super().__init__(root)

        self.cache = load_cache(INFERENCE_CACHE_NAME)
        if self.cache is None:
            self.cache = {
                "save_dir" : "data/runs/inference",
                "result_name" : "experiment001",
                "target" : "",
                "model" : "",
                "use_segmentation" : False
            }

        self.open_progress_page = open_progress_page

        self.result_name_var = tk.StringVar(value=self.cache["result_name"])
        self.model_path_var = tk.StringVar(value=self.cache["model"])
        self.save_dir_var = tk.StringVar(value=self.cache["save_dir"])
        self.target_path_var = tk.StringVar(value=self.cache["target"])

        self.use_segmentation_var = tk.BooleanVar(value=self.cache["use_segmentation"])

        self.build(back_cmd)

    def open_model_file(self):
        file = askopenfile(mode ='r', filetypes =[('Model Files', '*.pt')])
        self.model_path_var.set(file.name)
    
    def open_img_file(self):
        file = askopenfile(mode ='r', filetypes =[('PNG Files', '*.png'), ('jpg Files', '*.jpg'), ('Video Files', '*.mp4')])
        self.target_path_var.set(file.name)

    def open_save_dir(self):
        dir_path = askdirectory()
        self.save_dir_var.set(dir_path)

    def save_cache(self):
        self.cache = {
            "save_dir" : self.save_dir_var.get(),
            "result_name" : self.result_name_var.get(),
            "target" : self.target_path_var.get(),
            "model" : self.model_path_var.get(),
            "use_segmentation" : self.use_segmentation_var.get()
        }
        save_cache(self.cache, INFERENCE_CACHE_NAME)

    def get_source_size(self):
        path = self.target_path_var.get()
        ext = os.path.splitext(path)[1].lower()
        # If is image
        if ext in [".png", ".jpg"]:
            size = Image.open(path).size
            return size
        elif ext in [".mp4"]: # else is a video
            scanner = VideoScanner(path)
            return scanner.get_frame_size()

    def run(self):
        save_path = os.path.join(self.save_dir_var.get(), self.result_name_var.get()) # NEED TO CHECK UNIQUE

        yolo_call = YOLO_Call(
            seg=self.use_segmentation_var.get(),
            train=False,
            devices=-1, # Need to be more flexible here
            weights=self.model_path_var.get(),
            project=self.save_dir_var.get(),
            name=self.result_name_var.get(),
            source=self.target_path_var.get(),
            img=min(self.get_source_size()),
            save_txt=True,
            no_trace=True      
        )

        self.save_cache()
        self.open_progress_page(yolo_call, save_path)

    def build(self, back_cmd=lambda: None):
        greeting = HeaderLabel(self, text="Inference Frame")
        greeting.pack()

        back_btn = MenuButton(self,
            text="Home Page",
            command=back_cmd
        )
        back_btn.pack()

        model_btn = ctk.CTkButton(self,
            text="Load Model",
            width=20,
            height=3,
            command=self.open_model_file
        )
        model_btn.pack(pady=4)

        model_path_entry = tk.Entry(self, textvariable=self.model_path_var, state="disabled")
        model_path_entry.pack(pady=4)

        load_img_btn = ctk.CTkButton(self,
            text="Load Image/Video",
            width=20,
            height=3,
            command=self.open_img_file
        )
        load_img_btn.pack(pady=4)
        

        target_path_entry = tk.Entry(self, textvariable=self.target_path_var, state="disabled")
        target_path_entry.pack(pady=4)

        save_dir_btn = ctk.CTkButton(self,
            text="Save Directory",
            width=20,
            height=3,
            command=self.open_save_dir
        )
        save_dir_btn.pack(pady=4)

        save_dir_entry = tk.Entry(self, textvariable=self.save_dir_var, state="disabled")
        save_dir_entry.pack(pady=4)

        result_lbl = ctk.CTkLabel(self, text="Result Name")
        result_lbl.pack(pady=4)
        result_name_entry = tk.Entry(self, textvariable=self.result_name_var)
        result_name_entry.pack(pady=4)

        seg_chkb = ttk.Checkbutton(self,
            text='Use segmentation',
            variable=self.use_segmentation_var,
            onvalue=True,
            offvalue=False
        )
        seg_chkb.pack(pady=4)

        run_btn = ctk.CTkButton(self,
            text="Run",
            width=50,
            height=10,
            fg_color="red",
            command=self.run
        )
        run_btn.pack(pady=4)





