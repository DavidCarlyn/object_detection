import tkinter as tk

class TrainingFrame(tk.Frame):
    def __init__(self, root, back_cmd=lambda: None):
        super().__init__(root)

        self.build(back_cmd)

    def build(self, back_cmd=lambda: None):
        greeting = tk.Label(self, text="Training Frame")
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

