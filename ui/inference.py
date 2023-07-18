import os
import subprocess

import tkinter as tk
from tkinter.filedialog import askopenfile, askdirectory

from ui.buttons import MenuButton
from ui.labels import HeaderLabel

class InferenceFrame(tk.Frame):
    def __init__(self, root, back_cmd=lambda: None):
        super().__init__(root)
        self.result_name_var = tk.StringVar(value="experiment001")
        self.img_size_var = tk.StringVar(value="256")
        self.model_path_var = tk.StringVar(value="")
        self.save_dir_var = tk.StringVar(value="data/runs/inference")
        self.target_path_var = tk.StringVar(value="")

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

    def run(self):
        script_path = os.path.join(self.master.project_path, "externals", "yolov7", "detect.py")
        cmd_str = f"python {script_path} --source {self.target_path_var.get()}" 
        cmd_str += f" --weights {self.model_path_var.get()}"
        cmd_str += f" --project {self.save_dir_var.get()}"
        cmd_str += f" --name {self.result_name_var.get()}"
        cmd_str += f" --img {self.img_size_var.get()} --save-txt"

        print(cmd_str)

        stdout = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).stdout

        print(stdout.read())
        print("DONE")

    def build(self, back_cmd=lambda: None):
        greeting = HeaderLabel(self, text="Inference Frame")
        greeting.pack()

        back_btn = MenuButton(self,
            text="Home Page",
            command=back_cmd
        )
        back_btn.pack()

        model_btn = tk.Button(self,
            text="Load Model",
            width=20,
            height=3,
            bg="lightgrey",
            fg="black",
            command=self.open_model_file
        )
        model_btn.pack()

        model_path_entry = tk.Entry(self, textvariable=self.model_path_var, state="disabled")
        model_path_entry.pack()

        load_img_btn = tk.Button(self,
            text="Load Image/Video",
            width=20,
            height=3,
            bg="lightgrey",
            fg="black",
            command=self.open_img_file
        )
        load_img_btn.pack()

        target_path_entry = tk.Entry(self, textvariable=self.target_path_var, state="disabled")
        target_path_entry.pack()

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
        
        img_size_lbl = tk.Label(self, text="Image Size")
        img_size_lbl.pack()
        img_size_entry = tk.Entry(self, textvariable=self.img_size_var)
        img_size_entry.pack()

        run_btn = tk.Button(self,
            text="Run",
            width=50,
            height=10,
            bg="red",
            fg="white",
            command=self.run
        )
        run_btn.pack()





