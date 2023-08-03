import os

import tkinter as tk
import customtkinter as ctk
from ui.home import HomeFrame
from ui.inference import InferenceFrame
from ui.training import TrainingFrame

ctk.set_appearance_mode("dark")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Rust Detector Management")
        self.geometry('500x500')
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        self.build_home_window()

    def clear_window(self):
        for child in self.winfo_children():
            child.destroy()

    def build_train_window(self):
        self.clear_window()
        frame = TrainingFrame(self, back_cmd=self.build_home_window)
        frame.pack()

    def build_infer_window(self):
        self.clear_window()
        frame = InferenceFrame(self, back_cmd=self.build_home_window)
        frame.pack()

    def build_home_window(self):
        self.clear_window()
        home_page = HomeFrame(self,
            train_cmd=self.build_train_window,
            infer_cmd=self.build_infer_window
        )
        home_page.pack()

    def __str__(self) -> str:
        return "Main App"

if __name__ == "__main__":
    app = App()
    app.mainloop()


