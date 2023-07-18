import tkinter as tk

class MenuButton(tk.Button):
    def __init__(self, root, text="Button", bg="lightgrey", fg="black", font=("Arial", 16), **kwargs) -> None:
        super().__init__(root, text=text, bg=bg, fg=fg, font=font, **kwargs)

    def pack(self, ipadx=5, ipady=5, expand=True, **kwargs):
        super().pack(ipadx=ipadx, ipady=ipady, expand=expand, **kwargs)
    
    def grid(self, ipadx=5, ipady=5, expand=True, **kwargs):
        super().pack(ipadx=ipadx, ipady=ipady, expand=expand, **kwargs)
