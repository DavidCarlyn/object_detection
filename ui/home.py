import tkinter as tk

from ui.buttons import MenuButton
from ui.labels import HeaderLabel

class HomeFrame(tk.Frame):
    def __init__(self, root, train_cmd=lambda: None, infer_cmd=lambda: None):
        super().__init__(root)

        self.build(train_cmd, infer_cmd)

    def build(self, train_cmd=lambda: None, infer_cmd=lambda: None):
        greeting = HeaderLabel(self, text="Rust Detector Management")
        greeting.grid(row=0, column=0)

        button_frame = tk.Frame(self)
        button_frame.grid(row=1, column=0)

        train_btn = MenuButton(button_frame,
            text="Create Model",
            command=train_cmd
        )
        train_btn.pack(side=tk.LEFT, padx=20, pady=20)
        
        infer_btn = MenuButton(button_frame,
            text="Use Model",
            command=infer_cmd
        )
        infer_btn.pack(side=tk.RIGHT, padx=20, pady=20)

