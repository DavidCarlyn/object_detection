import tkinter as tk
import customtkinter as ctk
from ui.buttons import MenuButton
from ui.labels import HeaderLabel

class HomeFrame(ctk.CTkFrame):
    def __init__(self, root, train_cmd=lambda: None, infer_cmd=lambda: None, anno_cmd=lambda: None):
        super().__init__(root)
        self.build(train_cmd, infer_cmd, anno_cmd)

    def build(self, train_cmd=lambda: None, infer_cmd=lambda: None, anno_cmd=lambda: None):
        greeting = HeaderLabel(self, text="Rust Detector Management")
        greeting.grid(row=0, column=0)

        button_frame = ctk.CTkFrame(self)
        button_frame.grid(row=1, column=0)

        train_btn = MenuButton(button_frame,
            text="Create Model",
            command=train_cmd
        )
        train_btn.pack(side=ctk.LEFT, padx=20, pady=20)
        
        infer_btn = MenuButton(button_frame,
            text="Use Model",
            command=infer_cmd
        )
        infer_btn.pack(side=ctk.RIGHT, padx=20, pady=20)
        
        anno_btn = MenuButton(button_frame,
            text="Convert\nAnnotations",
            command=anno_cmd
        )
        anno_btn.pack(side=ctk.RIGHT, padx=20, pady=20)

