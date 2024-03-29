import tkinter as tk
import customtkinter as ctk

class HeaderLabel(ctk.CTkLabel):
    def __init__(self, root, font=("Arial", 25), **kwargs):
        super().__init__(root, text_color="white", font=font, **kwargs)
        
    def pack(self, fill="x", side="top", pady=5, padx=5, ipadx=15, ipady=15, **kwargs):
        super().pack(fill=fill, side=side, pady=pady, padx=padx, ipadx=ipadx, ipady=ipady, **kwargs)
    
    def grid(self, pady=5, padx=5, ipadx=15, ipady=15, **kwargs):
        super().grid(pady=pady, padx=padx, ipadx=ipadx, ipady=ipady, **kwargs)

