from enum import Enum

import tkinter as tk
from tkinter import ttk
import customtkinter as ctk

from tkinter.filedialog import askopenfile, askdirectory

from ui.buttons import MenuButton
from ui.labels import HeaderLabel

class TrainingState(Enum):
    START = 0,
    TRAINING = 1,
    COMPLETED = 2,
    CANCELLED = 3
    
class TrainingProgressFrame(ctk.CTkFrame):
    def __init__(self, root, back_cmd=lambda: None, stop_thread_cmd=lambda: None):
        super().__init__(root)

        self.state = TrainingState.START
        self.stop_thread_cmd = stop_thread_cmd

        self.build(back_cmd)

    def is_training_ended(self):
        return self.state == TrainingState.CANCELLED or self.state == TrainingState.COMPLETED

    def start_training(self):
        self.change_state(TrainingState.TRAINING)

    def update_progress(self, cur_epoch, total_epoch):
        self.progress_lbl['text'] = f"Epoch {cur_epoch} of {total_epoch}"
        percent_complete = (cur_epoch / total_epoch)
        cur_prog = percent_complete * 100
        self.progress_bar['value'] = cur_prog

        if percent_complete >= 1.0:
            self.change_state(TrainingState.COMPLETED)

    def stop_run(self):
        print("STOP IN HERE")
        self.change_state(TrainingState.CANCELLED)
        self.stop_thread_cmd()

    def change_state(self, state):
        old_state = self.state
        self.state = state

        if state == TrainingState.TRAINING:
            self.progress_lbl = tk.Label(self,
                text="Starting up the training."
            )
            self.progress_lbl.pack(pady=4)
            self.progress_bar = ttk.Progressbar(self,
                orient='horizontal',
                mode='determinate',
                length=300,
            )
            self.progress_bar.pack(pady=8)
            self.stop_btn = ctk.CTkButton(self,
                text="Stop",
                width=20,
                height=3,
                fg_color="red",
                command=lambda: self.stop_run(),
            )
            self.stop_btn.pack(pady=4)

            self.back_btn["state"] = "disabled"
        elif state == TrainingState.CANCELLED:
            self.stop_btn.destroy()
            self.progress_bar.destroy()
            self.progress_lbl.destroy()
            end_lbl = HeaderLabel(self, text="Training Cancelled")
            end_lbl.pack(pady=4)
            self.back_btn["state"] = "normal"
        elif old_state != TrainingState.CANCELLED and state == TrainingState.COMPLETED:
            self.stop_btn.destroy()
            end_lbl = HeaderLabel(self, text="Training Completed!")
            end_lbl.pack(pady=4)
            self.back_btn["state"] = "normal"

    def build(self, back_cmd=lambda: None):
        greeting = HeaderLabel(self, text="Training Progress")
        greeting.pack()

        self.back_btn = MenuButton(self,
            text="Training Configuration Page",
            command=back_cmd
        )
        self.back_btn.pack(pady=4)
