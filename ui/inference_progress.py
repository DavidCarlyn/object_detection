from enum import Enum

import tkinter as tk
from tkinter import ttk

from tkinter.filedialog import askopenfile, askdirectory

from ui.buttons import MenuButton
from ui.labels import HeaderLabel

class InferenceState(Enum):
    START = 0,
    INFERRING = 1,
    COMPLETED = 2
    
class InferenceProgressFrame(tk.Frame):
    def __init__(self, root, back_cmd=lambda: None):
        super().__init__(root)

        self.state = InferenceState.START

        self.build(back_cmd)

    def is_inference_ended(self):
        return self.state == InferenceState.COMPLETED

    def start_inference(self):
        self.change_state(InferenceState.INFERRING)

    def update_progress(self, cur_epoch, total_epoch):
        self.progress_lbl['text'] = f"Frame {cur_epoch} of {total_epoch}"
        percent_complete = (cur_epoch / total_epoch)
        cur_prog = percent_complete * 100
        self.progress_bar['value'] = cur_prog

        if percent_complete >= 1.0:
            self.change_state(InferenceState.COMPLETED)

    def change_state(self, state):
        old_state = self.state
        self.state = state

        if state == InferenceState.INFERRING:
            self.progress_lbl = tk.Label(self,
                text="Starting up the inference."
            )
            self.progress_lbl.pack()
            self.progress_bar = ttk.Progressbar(self,
                orient='horizontal',
                mode='determinate',
                length=300,
            )
            self.progress_bar.pack()
            self.back_btn["state"] = "disabled"
        elif state == InferenceState.COMPLETED:
            end_lbl = HeaderLabel(self, text="Inference Completed!")
            end_lbl.pack()
            self.back_btn["state"] = "normal"

    def build(self, back_cmd=lambda: None):
        greeting = HeaderLabel(self, text="Inference Progress")
        greeting.pack()

        self.back_btn = MenuButton(self,
            text="Inference Configuration Page",
            command=back_cmd
        )
        self.back_btn.pack()
