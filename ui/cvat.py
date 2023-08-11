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
from processing.convert import convert


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
                "save_dir" : "data/runs/cvat",
                "result_name" : "experiment001",
                "target" : "",
            }

        self.result_name_var = tk.StringVar(value=self.cache["result_name"])
        self.cvat_path_var = tk.StringVar(value=self.cache["target"])
        self.save_dir_var = tk.StringVar(value=self.cache["save_dir"])
        self.build(back_cmd)

    def open_cvat_file(self):
        cfile = askopenfile(mode ='r', filetypes =[('XML', '*.xml')])
        self.cvat_path_var.set(cfile.name)

    def open_save_dir(self):
        dir_path = askdirectory()
        self.save_dir_var.set(dir_path)

    def save_cache(self):
        self.cache = {
            "save_dir" : self.save_dir_var.get(),
            "result_name" : self.result_name_var.get(),
            "target" : self.cvat_path_var.get(),
        }
        save_cache(self.cache, CVAT_CACHE_NAME)

    def run(self):
        save_path = os.path.join(self.save_dir_var.get(), self.result_name_var.get()) # NEED TO CHECK UNIQUE
        self.save_cache()

        os.makedirs(save_path, exist_ok=True)
        convert("cvat", "yolo", self.cvat_path_var.get(), "", os.path.join(self.save_dir_var.get(), self.result_name_var.get()))

        print("HI")

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
        
        save_dir_btn = ctk.CTkButton(self,
            text="Save Directory",
            width=20,
            height=3,
            command=self.open_save_dir
        )
        save_dir_btn.pack(pady=4)

        save_dir_entry = tk.Entry(self, textvariable=self.save_dir_var, state="disabled")
        save_dir_entry.pack(pady=4)

        run_btn = ctk.CTkButton(self,
            text="Run",
            width=50,
            height=10,
            fg_color="red",
            command=self.run
        )
        run_btn.pack(pady=4)

        