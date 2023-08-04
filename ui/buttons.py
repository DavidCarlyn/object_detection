import customtkinter as ctk

class MenuButton(ctk.CTkButton):
    def __init__(self, root, text="Button", bg="lightgrey", fg="black", font=("Arial", 16), **kwargs) -> None:
        super().__init__(root, text=text, font=font, **kwargs)

    def pack(self, ipadx=5, ipady=5, pady=5, padx=5, expand=True, **kwargs):
        super().pack(ipadx=ipadx, ipady=ipady, pady=pady, padx=padx, expand=expand, **kwargs)
    
    def grid(self, ipadx=5, ipady=5, pady=5, padx=5, expand=True, **kwargs):
        super().pack(ipadx=ipadx, ipady=ipady, pady=pady, padx=padx, expand=expand, **kwargs)
