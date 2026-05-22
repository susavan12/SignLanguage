import os
import subprocess
import sys
import threading

# Fix for Windows Python installs where Tcl/Tk is present but tkinter cannot find init.tcl.
if sys.platform == 'win32':
    base_prefix = getattr(sys, 'base_prefix', sys.prefix)
    tcl_root = os.path.join(base_prefix, 'tcl')
    tcl86 = os.path.join(tcl_root, 'tcl8.6')
    tk86 = os.path.join(tcl_root, 'tk8.6')
    if os.path.isdir(tcl86):
        os.environ.setdefault('TCL_LIBRARY', tcl86)
    if os.path.isdir(tk86):
        os.environ.setdefault('TK_LIBRARY', tk86)

import tkinter as tk
from tkinter import messagebox, simpledialog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
DATA_DIR = os.path.join(BASE_DIR, 'datasets')


def run_script(script_name, args=None):
    if args is None:
        args = []
    path = os.path.join(BASE_DIR, script_name)
    command = [PYTHON, path] + args

    def target():
        try:
            subprocess.run(command, cwd=BASE_DIR)
        except Exception as exc:
            messagebox.showerror('Error', f'Failed to run {script_name}: {exc}')

    threading.Thread(target=target, daemon=True).start()


def collect_data():
    label = simpledialog.askstring('Collect Data', 'Enter the sign label (e.g. A, B, C):')
    if not label:
        return

    count = simpledialog.askinteger('Collect Data', 'Number of images to capture:', initialvalue=200, minvalue=10, maxvalue=2000)
    if not count:
        return

    run_script('collect_data.py', ['--label', label.strip(), '--count', str(count), '--output-dir', DATA_DIR])


def train_model():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    run_script('train_model.py')


def run_detector():
    run_script('app.py')


def prepare_folders():
    labels = simpledialog.askstring('Prepare Folders', 'Enter labels separated by commas (e.g. A,B,C):')
    if not labels:
        return

    missing = []
    for raw_label in labels.split(','):
        label = raw_label.strip()
        if not label:
            continue
        folder = os.path.join(DATA_DIR, label)
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as exc:
            missing.append(label)
            print(f'Could not create folder for {label}: {exc}')

    if missing:
        messagebox.showwarning('Warning', f'Could not create folders for: {", ".join(missing)}')
    else:
        messagebox.showinfo('Done', 'Dataset folders are ready.')


def build_gui():
    root = tk.Tk()
    root.title('Sign Language Detection')
    root.geometry('400x260')
    root.resizable(False, False)

    title = tk.Label(root, text='Sign Language Detection', font=('Arial', 16, 'bold'))
    title.pack(pady=(16, 8))

    description = tk.Label(root, text='Use this GUI to prepare data, train the model, and run detection.', wraplength=360, justify='center')
    description.pack(pady=(0, 16))

    btn_prepare = tk.Button(root, text='Create Label Folders', width=28, command=prepare_folders)
    btn_prepare.pack(pady=6)

    btn_collect = tk.Button(root, text='Collect Sign Images', width=28, command=collect_data)
    btn_collect.pack(pady=6)

    btn_train = tk.Button(root, text='Train Model', width=28, command=train_model)
    btn_train.pack(pady=6)

    btn_run = tk.Button(root, text='Run Detector', width=28, command=run_detector)
    btn_run.pack(pady=6)

    footer = tk.Label(root, text='Press Q in the camera window to stop collection or detection.', font=('Arial', 9))
    footer.pack(pady=(12, 0))

    root.mainloop()


if __name__ == '__main__':
    build_gui()