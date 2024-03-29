import os
import subprocess

import tkinter as tk
import customtkinter as ctk
from tkinter import ttk

from tkinter.filedialog import askopenfile, askdirectory

from PIL import Image

from ui.buttons import MenuButton
from ui.labels import HeaderLabel
from processing.utils import load_cache, save_cache, get_available_gpus, is_on_windows, load_yaml
from processing.yolo_api import YOLO_Call

TRAINING_CACHE_NAME = "training.cache"
class TrainingFrame(ctk.CTkFrame):
    def __init__(self, root, back_cmd=lambda: None, open_progress_page=lambda: None):
        super().__init__(root)

        self.cache = load_cache(TRAINING_CACHE_NAME)
        if self.cache is None:
            self.cache = {
                "save_dir" : "data/runs/training",
                "result_name" : "experiment001",
                "epochs" : 300,
                "workers" : 2,
                "batch_size" : 2,
                "data_file" : "",
                "model_config" : "",
                "model" : "",
                "training_config" : "",
                "use_gpu" : False,
                "use_segmentation" : False
            }

        self.save_dir_var = tk.StringVar(value=self.cache["save_dir"])
        self.result_name_var = tk.StringVar(value=self.cache["result_name"])

        self.epochs_var = tk.IntVar(value=self.cache["epochs"])
        self.workers_var = tk.IntVar(value=self.cache["workers"])
        self.batch_size_var = tk.IntVar(value=self.cache["batch_size"])

        self.data_file_var = tk.StringVar(value=self.cache["data_file"])
        self.model_config = tk.StringVar(value=self.cache["model_config"])
        self.model_var = tk.StringVar(value=self.cache["model"])
        self.training_config_var = tk.StringVar(value=self.cache["training_config"])

        self.use_gpu_var = tk.BooleanVar(value=self.cache["use_gpu"])
        self.use_segmentation_var = tk.BooleanVar(value=self.cache["use_segmentation"])

        self.workers_var = tk.IntVar(value=8)
        self.batch_size_var = tk.IntVar(value=4)
        self.error_lbl = None

        self.open_progress_page = open_progress_page

        self.build(back_cmd)

    def open_save_dir(self):
        dir_path = askdirectory()
        self.save_dir_var.set(dir_path)

    def open_model_file(self):
        file = askopenfile(mode ='r', filetypes =[('Model Files', '*.pt')])
        self.model_var.set(file.name)
    
    def open_yaml_file(self, ent_var):
        file = askopenfile(mode ='r', filetypes =[('YAML Files', '*.yaml'), ('YAML Files', '*.yml')])
        ent_var.set(file.name)

    def save_cache(self):
        self.cache = {
            "save_dir" : self.save_dir_var.get(),
            "result_name" : self.result_name_var.get(),
            "epochs" : self.epochs_var.get(),
            "workers" : self.workers_var.get(),
            "batch_size" : self.batch_size_var.get(),
            "data_file" : self.data_file_var.get(),
            "model_config" : self.model_config.get(),
            "model" : self.model_var.get(),
            "training_config" : self.training_config_var.get(),
            "use_gpu" : self.use_gpu_var.get(),
            "use_segmentation" : self.use_segmentation_var.get()
        }
        save_cache(self.cache, TRAINING_CACHE_NAME)

    def add_exists_error(self, path):
        if self.error_lbl is not None:
            return
        
        self.error_lbl = tk.Label(self,
            text=f"ERROR: {path} already exists. Please choose a different 'Result Name' or 'Save Directory'.",
            bg="yellow",
            fg="red"
        )
        self.error_lbl.pack()

    def get_image_size(self):
        data_file = load_yaml(os.path.join(self.data_file_var.get()))

        ex_path = None
        for root, dirs, files in os.walk(data_file['train']):
            for f in files:
                if os.path.splitext(f)[1].lower() in [".png", ".jpg"]:
                    ex_path = os.path.join(root, f)
                    break
            if ex_path is not None: break
        if ex_path is None:
            return None
        
        size = Image.open(ex_path).size
        return size


    def run(self):
        save_path = os.path.join(self.save_dir_var.get(), self.result_name_var.get())
        if os.path.exists(save_path):
            self.add_exists_error(save_path)
            return

        devices = "-1"
        if self.use_gpu_var.get():
            devices = ",".join(get_available_gpus())
        
        yolo_call = YOLO_Call(
            seg=self.use_segmentation_var.get(),
            train=True,
            devices=devices,
            workers=self.workers_var.get(),
            batch_size=self.batch_size_var.get(),
            img=min(self.get_image_size()),
            epochs=self.epochs_var.get(),
            project=self.save_dir_var.get(),
            name=self.result_name_var.get(),
            data=self.data_file_var.get(),
            cfg=self.model_config.get(),
            weights=self.model_var.get(),
            hyp=self.training_config_var.get()
        )

        self.save_cache()
        self.open_progress_page(yolo_call, save_path, total_epochs=self.epochs_var.get())

    def build(self, back_cmd=lambda: None):
        greeting = HeaderLabel(self, text="Training Frame")
        greeting.pack()

        back_btn = MenuButton(self,
            text="Home Page",
            command=back_cmd
        )
        back_btn.pack()

        save_dir_btn = ctk.CTkButton(self,
            text="Save Directory",
            width=20,
            height=3,
            command=self.open_save_dir
        )
        save_dir_btn.pack()

        save_dir_entry = tk.Entry(self, textvariable=self.save_dir_var, state="disabled")
        save_dir_entry.pack(pady=4)

        result_lbl = ctk.CTkLabel(self, text="Result Name")
        result_lbl.pack(pady=4)
        result_name_entry = tk.Entry(self, textvariable=self.result_name_var)
        result_name_entry.pack(pady=4)

        worker_lbl = ctk.CTkLabel(self, text="Number of workers")
        worker_lbl.pack(pady=4)
        worker_entry = tk.Entry(self, textvariable=self.workers_var)
        worker_entry.pack(pady=4)
        
        batch_size_lbl = ctk.CTkLabel(self, text="Batch size")
        batch_size_lbl.pack(pady=4)
        batch_size_entry = tk.Entry(self, textvariable=self.batch_size_var)
        batch_size_entry.pack(pady=4)
        
        epochs_lbl = tk.Label(self, text="Number of Epochs")
        epochs_lbl.pack(pady=4)
        epochs_entry = tk.Entry(self, textvariable=self.epochs_var)
        epochs_entry.pack(pady=4)

        gpu_chkb = ttk.Checkbutton(self,
            text='Use GPUs',
            variable=self.use_gpu_var,
            onvalue=True,
            offvalue=False
        )
        gpu_chkb.pack(pady=4)
        
        seg_chkb = ttk.Checkbutton(self,
            text='Train segmentation',
            variable=self.use_segmentation_var,
            onvalue=True,
            offvalue=False
        )
        seg_chkb.pack(pady=4)

        data_file_btn = ctk.CTkButton(self,
            text="Data file",
            width=20,
            height=3,
            command=lambda: self.open_yaml_file(self.data_file_var)
        )
        data_file_btn.pack(pady=4)

        data_file_entry = tk.Entry(self, textvariable=self.data_file_var, state="disabled")
        data_file_entry.pack(pady=4)
        
        model_config_file_btn = ctk.CTkButton(self,
            text="Model Configuration File",
            width=20,
            height=3,
            command=lambda: self.open_yaml_file(self.model_config)
        )
        model_config_file_btn.pack(pady=4)

        model_config_file_entry = tk.Entry(self, textvariable=self.model_config, state="disabled")
        model_config_file_entry.pack(pady=4)
        
        model_weights_file_btn = ctk.CTkButton(self,
            text="Model Weights File",
            width=20,
            height=3,
            command=lambda: self.open_model_file()
        )
        model_weights_file_btn.pack(pady=4)

        model_weights_file_entry = tk.Entry(self, textvariable=self.model_var, state="disabled")
        model_weights_file_entry.pack(pady=4)
        
        training_file_btn = ctk.CTkButton(self,
            text="Training Configuration File",
            width=20,
            height=3,
            command=lambda: self.open_yaml_file(self.training_config_var)
        )
        training_file_btn.pack(pady=4)

        training_file_entry = tk.Entry(self, textvariable=self.training_config_var, state="disabled")
        training_file_entry.pack(pady=4)
        
        run_btn = ctk.CTkButton(self,
            text="Run",
            width=50,
            height=10,
            command=self.run,
            fg_color="red"
        )
        run_btn.pack(pady=4)

