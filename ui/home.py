import tkinter as tk

class HomeFrame(tk.Frame):
    def __init__(self, root, train_cmd=lambda: None, infer_cmd=lambda: None):
        super().__init__(root)

        self.build(train_cmd, infer_cmd)

    def build(self, train_cmd=lambda: None, infer_cmd=lambda: None):
        greeting = tk.Label(self, text="Rust Detector Management")
        greeting.pack()

        train_btn = tk.Button(self,
            text="Create Model",
            width=50,
            height=10,
            bg="lightgrey",
            fg="black",
            command=train_cmd
        )
        train_btn.pack()
        
        infer_btn = tk.Button(self,
            text="Use Model",
            width=50,
            height=10,
            bg="lightgrey",
            fg="black",
            command=infer_cmd
        )
        infer_btn.pack()

