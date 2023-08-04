import os
import multiprocessing as mp
import threading

import tkinter as tk
import customtkinter as ctk

from ui.home import HomeFrame
from ui.inference import InferenceFrame
from ui.training import TrainingFrame
from ui.training_progress import TrainingProgressFrame
from ui.inference_progress import InferenceProgressFrame

from processing.utils import execute_command

ctk.set_appearance_mode("dark")

"""
Main class for training and using the YOLOv7 model
"""
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Rust Detector Management")
        #self.geometry('')
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        self.run_thread = None
        self.last_result = None
        
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
            print("stop start")
            self.run_thread.terminate()
            self.run_thread.join()

    def update_train_window(self, save_path, frame):
        if frame.is_training_ended():
            print("Training Ended")
            return

        results_path = os.path.join(save_path, "results.txt")
        if os.path.exists(results_path):
            with open(results_path) as f:
                lines = f.readlines()
                epoch_info = lines[-1].split()[0].split("/")
                if len(epoch_info) == 2:
                    cur_epoch = int(epoch_info[0])
                    total_epoch = int(epoch_info[1])
                    if cur_epoch != self.last_result:
                        frame.update_progress(cur_epoch+1, total_epoch+1)
                        self.last_result = cur_epoch

        self.after(1000, self.update_train_window, save_path, frame)
    
    def update_infer_window(self, save_path, frame):
        if frame.is_inference_ended():
            print("Inference Ended")
            return

        while self.process_conn.poll():            
            line = self.process_conn.recv()
            if 'video' in line:
                cur, end = line.split()[2].split('/')
                cur = int(cur[1:])
                end = int(end[:-1])
                frame.update_progress(cur, end)

        self.after(1000, self.update_infer_window, save_path, frame)
        
    def build_train_progress_window(self, yolo_call_obj, save_path):
        self.clear_window()
        frame = TrainingProgressFrame(self, back_cmd=self.build_train_window, stop_thread_cmd=self.stop_thread)
        frame.pack()
        frame.start_training()

        self.run_thread = mp.Process(target=yolo_call_obj.call)
        self.run_thread.start()

        self.update_train_window(save_path, frame)
    
    def build_infer_progress_window(self, yolo_call_obj, save_path):
        self.clear_window()
        frame = InferenceProgressFrame(self, back_cmd=self.build_infer_window)
        frame.pack()
        frame.start_inference()

        self.process_conn, child_conn = mp.Pipe(duplex=True)
        self.run_thread = mp.Process(target=yolo_call_obj.call, args=(child_conn, ))
        self.run_thread.start()

        self.update_infer_window(save_path, frame)

    def build_infer_window(self):
        self.clear_window()
        frame = InferenceFrame(self, back_cmd=self.build_home_window, open_progress_page=self.build_infer_progress_window)
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
    print("Loading Object Detection Application.")
    print("This may take a minute...")
    mp.freeze_support()
    app = App()
    app.mainloop()
    if app.run_thread is not None:
        app.run_thread.terminate()


