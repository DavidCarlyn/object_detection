import os
import multiprocessing as mp
import threading

import tkinter as tk

from ui.home import HomeFrame
from ui.inference import InferenceFrame
from ui.training import TrainingFrame
from ui.training_progress import TrainingProgressFrame

from utils import execute_command

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rust Detector Management")
        self.geometry('500x500')
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        self.run_thread = None
        
        self.build_home_window()

    def clear_window(self):
        for child in self.winfo_children():
            child.destroy()

    def build_train_window(self):
        self.clear_window()
        frame = TrainingFrame(self, back_cmd=self.build_home_window, open_progress_page=self.build_train_progress_window)
        frame.pack()

    def stop_thread(self):
        if self.run_thread is not None:
            self.run_thread.terminate()

    def update_train_window(self, conn):
        if conn.closed:
            print("THREAD CLOSED")
            return
        print("="*100)
        print("TEST LINE")

        if conn.poll():
            data = conn.recv()
            print(data)

        threading.Timer(1, self.update_train_window, conn).start()
        
    def build_train_progress_window(self, cmd_str):
        self.clear_window()
        frame = TrainingProgressFrame(self, back_cmd=self.build_train_window, stop_thread_cmd=self.stop_thread)
        frame.pack()

        parent_conn, child_conn = mp.Pipe()
        self.run_thread = mp.Process(target=execute_command, args=(cmd_str, child_conn))
        self.run_thread.start()

        self.update_train_window(parent_conn)



    
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
    if app.run_thread is not None:
        app.run_thread.terminate()


