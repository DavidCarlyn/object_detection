import tkinter as tk
from tkinter.filedialog import askopenfile

class InferenceFrame(tk.Frame):
    def __init__(self, root, back_cmd=lambda: None):
        super().__init__(root)
        self.result_name_var = tk.StringVar(value="experiment001")
        self.img_size_var = tk.StringVar(value="256")

        self.build(back_cmd)

    def open_model_file(self):
        file = askopenfile(mode ='r', filetypes =[('Model Files', '*.pt')])
        print(file.name)
    
    def open_img_file(self):
        file = askopenfile(mode ='r', filetypes =[('PNG Files', '*.png'), ('jpg Files', '*.jpg'), ('Video Files', '*.mp4')])
        print(file.name)

    def run(self):
        print(self.result_name_var.get())
        print(self.img_size_var.get())

    def build(self, back_cmd=lambda: None):
        greeting = tk.Label(self, text="Inference Frame")
        greeting.pack()

        back_btn = tk.Button(self,
            text="Home Page",
            width=20,
            height=5,
            bg="lightgrey",
            fg="black",
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

        load_img_btn = tk.Button(self,
            text="Load Image/Video",
            width=20,
            height=3,
            bg="lightgrey",
            fg="black",
            command=self.open_img_file
        )
        load_img_btn.pack()

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





