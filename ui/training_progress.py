from enum import Enum

import tkinter as tk

from tkinter.filedialog import askopenfile, askdirectory

from ui.buttons import MenuButton
from ui.labels import HeaderLabel

class TrainingState(Enum):
    START = 0,
    TRAINING = 1,
    COMPLETED = 2,
    CANCELLED = 3
    
class TrainingProgressFrame(tk.Frame):
    def __init__(self, root, back_cmd=lambda: None, stop_thread_cmd=lambda: None):
        super().__init__(root)

        self.state = TrainingState.START
        self.stop_thread_cmd = stop_thread_cmd

        self.build(back_cmd)

    def run(self, cmd_str):
        print(cmd_str)

        self.change_state(TrainingState.TRAINING)

        self.change_state(TrainingState.COMPLETED)

    def stop_run(self):
        self.change_state(TrainingState.CANCELLED)
        self.stop_thread_cmd()

    def change_state(self, state):
        old_state = self.state
        self.state = state

        if state == TrainingState.TRAINING:
            self.stop_btn = tk.Button(self,
                text="Stop",
                width=20,
                height=3,
                bg="red",
                fg="white",
                command=lambda: self.stop_run
            )

            self.stop_btn.pack()
            self.back_btn["state"] = "disabled"
        elif state == TrainingState.CANCELLED:
            self.stop_btn.destroy()
            end_lbl = HeaderLabel(self, text="Training Cancelled")
            end_lbl.pack()
            self.back_btn["state"] = "normal"
        elif old_state != TrainingState.CANCELLED and state == TrainingState.COMPLETED:
            self.stop_btn.destroy()
            end_lbl = HeaderLabel(self, text="Training Completed!")
            end_lbl.pack()
            self.back_btn["state"] = "normal"

    def build(self, back_cmd=lambda: None):
        greeting = HeaderLabel(self, text="Training Progress")
        greeting.pack()

        self.back_btn = MenuButton(self,
            text="Training Configuration Page",
            command=back_cmd
        )
        self.back_btn.pack()
