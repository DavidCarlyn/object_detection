import os
import shutil
import subprocess

import tkinter as tk
import customtkinter as ctk
from tkinter import ttk
from tkinter.filedialog import askopenfile, askdirectory

from ui.buttons import MenuButton
from ui.labels import HeaderLabel
from processing.utils import load_cache, save_cache, VideoScanner
from processing.yolo_api import YOLO_Call
from processing.convert import json2yolo


"""
You can just 
1. copy the inference page frame class, 
2. Edit names, 
3. remove unecessary fields. 

It should only have a SAVE DIRECTORY and an upload for the CVAT ANNOTATIONS.

"""

CVAT_CACHE_NAME = "cvat.cache"
class CVATFrame(ctk.CTkFrame):
    def __init__(self, root, back_cmd=lambda: None):
        super().__init__(root)

        self.cache = load_cache(CVAT_CACHE_NAME)
        if self.cache is None:
            self.cache = {
                "save_dir" : "data/runs/annotations",
                "result_name" : "annotation_01",
                "target" : "",
                "use_segmentation" : False,
            }

        self.result_name_var = tk.StringVar(value=self.cache["result_name"])
        self.cvat_path_var = tk.StringVar(value=self.cache["target"])
        self.save_dir_var = tk.StringVar(value=self.cache["save_dir"])
        self.use_segmentation_var = tk.BooleanVar(value=self.cache["use_segmentation"])

        self.error_lbl = None

        self.build(back_cmd)

    def open_cvat_file(self):
        dir_path = askdirectory()
        self.cvat_path_var.set(dir_path)

    def open_save_dir(self):
        dir_path = askdirectory()
        self.save_dir_var.set(dir_path)

    def save_cache(self):
        self.cache = {
            "save_dir" : self.save_dir_var.get(),
            "result_name" : self.result_name_var.get(),
            "target" : self.cvat_path_var.get(),
            "use_segmentation" : self.use_segmentation_var.get(),
        }
        save_cache(self.cache, CVAT_CACHE_NAME)

    def add_exists_error(self, path):
        if self.error_lbl is not None:
            return
        
        self.error_lbl = tk.Label(self,
            text=f"ERROR: {path} already exists. Please choose a different 'Result Name' or 'Save Directory'.",
            bg="yellow",
            fg="red"
        )
        self.error_lbl.pack()

    def run(self):
        save_path = os.path.join(self.save_dir_var.get(), self.result_name_var.get())
        if os.path.exists(save_path):
            self.add_exists_error(save_path)
            return
        self.save_cache()

        os.makedirs(save_path, exist_ok=True)

        json2yolo(self.cvat_path_var.get(), use_segments=self.use_segmentation_var.get(), save_dir=save_path)
        lbl_dir = os.path.join(save_path, "labels", "default")
        for fname in os.listdir(lbl_dir):
            shutil.move(os.path.join(lbl_dir, fname), os.path.join(save_path, "labels", fname))
        os.rmdir(lbl_dir)

    def build(self, back_cmd=lambda: None):
        greeting = HeaderLabel(self, text="Inference Frame")
        greeting.pack()

        back_btn = MenuButton(self,
            text="Home Page",
            command=back_cmd
        )
        back_btn.pack()

        model_btn = ctk.CTkButton(self,
            text="Load Annotations",
            width=20,
            height=3,
            command=self.open_cvat_file
        )
        model_btn.pack(pady=4)

        cvat_dir_entry = tk.Entry(self, textvariable=self.cvat_path_var, state="disabled")
        cvat_dir_entry.pack(pady=4)
        
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

        