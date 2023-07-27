import os
import subprocess

import tkinter as tk

from tkinter.filedialog import askopenfile, askdirectory

from ui.buttons import MenuButton
from ui.labels import HeaderLabel
from utils import load_cache, save_cache

TRAINING_CACHE_NAME = "training.cache"
class TrainingFrame(tk.Frame):
    def __init__(self, root, back_cmd=lambda: None, open_progress_page=lambda: None):
        super().__init__(root)

        self.cache = load_cache(TRAINING_CACHE_NAME)
        if self.cache is None:
            self.cache = {
                "save_dir" : "data/runs/training",
                "result_name" : "experiment001",
                "img_size" : 512,
                "epochs" : 300,
                "workers" : 2,
                "batch_size" : 2,
                "data_file" : "",
                "model_config" : "",
                "model" : "",
                "training_config" : ""
            }

        self.save_dir_var = tk.StringVar(value=self.cache["save_dir"])
        self.result_name_var = tk.StringVar(value=self.cache["result_name"])

        self.img_size_var = tk.IntVar(value=self.cache["img_size"])
        self.epochs_var = tk.IntVar(value=self.cache["epochs"])
        self.workers_var = tk.IntVar(value=self.cache["workers"])
        self.batch_size_var = tk.IntVar(value=self.cache["batch_size"])

        self.data_file_var = tk.StringVar(value=self.cache["data_file"])
        self.model_config = tk.StringVar(value=self.cache["model_config"])
        self.model_var = tk.StringVar(value=self.cache["model"])
        self.training_config_var = tk.StringVar(value=self.cache["training_config"])

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
            "img_size" : self.img_size_var.get(),
            "epochs" : self.epochs_var.get(),
            "workers" : self.workers_var.get(),
            "batch_size" : self.batch_size_var.get(),
            "data_file" : self.data_file_var.get(),
            "model_config" : self.model_config.get(),
            "model" : self.model_var.get(),
            "training_config" : self.training_config_var.get()
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

    def run(self):
        save_path = os.path.join(self.save_dir_var.get(), self.result_name_var.get())
        if os.path.exists(save_path):
            self.add_exists_error(save_path)
            return

        script_path = os.path.join(self.master.project_path, "externals", "yolov7", "train.py")
        cmd_str = f"python {script_path} --workers {self.workers_var.get()} --batch-size {self.batch_size_var.get()}" 
        cmd_str += f" --img {self.img_size_var.get()}"
        cmd_str += f" --epochs {self.epochs_var.get()}"
        cmd_str += f" --project {self.save_dir_var.get()}"
        cmd_str += f" --name {self.result_name_var.get()}"
        cmd_str += f" --data {self.data_file_var.get()}"
        cmd_str += f" --cfg {self.model_config.get()}"
        cmd_str += f" --weights {self.model_var.get()}"
        cmd_str += f" --hyp {self.training_config_var.get()}"

        self.save_cache()
        self.open_progress_page(cmd_str, save_path)

    def build(self, back_cmd=lambda: None):
        greeting = HeaderLabel(self, text="Training Frame")
        greeting.pack()

        back_btn = MenuButton(self,
            text="Home Page",
            command=back_cmd
        )
        back_btn.pack()

        save_dir_btn = tk.Button(self,
            text="Save Directory",
            width=20,
            height=3,
            bg="lightgrey",
            fg="black",
            command=self.open_save_dir
        )
        save_dir_btn.pack()

        save_dir_entry = tk.Entry(self, textvariable=self.save_dir_var, state="disabled")
        save_dir_entry.pack()

        result_lbl = tk.Label(self, text="Result Name")
        result_lbl.pack()
        result_name_entry = tk.Entry(self, textvariable=self.result_name_var)
        result_name_entry.pack()

        worker_lbl = tk.Label(self, text="Number of workers")
        worker_lbl.pack()
        worker_entry = tk.Entry(self, textvariable=self.workers_var)
        worker_entry.pack()
        
        batch_size_lbl = tk.Label(self, text="Batch size")
        batch_size_lbl.pack()
        batch_size_entry = tk.Entry(self, textvariable=self.batch_size_var)
        batch_size_entry.pack()

        img_size_lbl = tk.Label(self, text="Image size")
        img_size_lbl.pack()
        img_size_entry = tk.Entry(self, textvariable=self.img_size_var)
        img_size_entry.pack()
        
        epochs_lbl = tk.Label(self, text="Number of Epochs")
        epochs_lbl.pack()
        epochs_entry = tk.Entry(self, textvariable=self.epochs_var)
        epochs_entry.pack()

        data_file_btn = tk.Button(self,
            text="Data file",
            width=20,
            height=3,
            bg="lightgrey",
            fg="black",
            command=lambda: self.open_yaml_file(self.data_file_var)
        )
        data_file_btn.pack()

        data_file_entry = tk.Entry(self, textvariable=self.data_file_var, state="disabled")
        data_file_entry.pack()
        
        model_config_file_btn = tk.Button(self,
            text="Model Configuration File",
            width=20,
            height=3,
            bg="lightgrey",
            fg="black",
            command=lambda: self.open_yaml_file(self.model_config)
        )
        model_config_file_btn.pack()

        model_config_file_entry = tk.Entry(self, textvariable=self.model_config, state="disabled")
        model_config_file_entry.pack()
        
        model_weights_file_btn = tk.Button(self,
            text="Model Weights File",
            width=20,
            height=3,
            bg="lightgrey",
            fg="black",
            command=lambda: self.open_model_file()
        )
        model_weights_file_btn.pack()

        model_weights_file_entry = tk.Entry(self, textvariable=self.model_var, state="disabled")
        model_weights_file_entry.pack()
        
        training_file_btn = tk.Button(self,
            text="Training Configuration File",
            width=20,
            height=3,
            bg="lightgrey",
            fg="black",
            command=lambda: self.open_yaml_file(self.training_config_var)
        )
        training_file_btn.pack()

        training_file_entry = tk.Entry(self, textvariable=self.training_config_var, state="disabled")
        training_file_entry.pack()
        
        run_btn = tk.Button(self,
            text="Run",
            width=50,
            height=10,
            bg="red",
            fg="white",
            command=self.run
        )
        run_btn.pack()

