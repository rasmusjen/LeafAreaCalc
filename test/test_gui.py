# src/gui_leaf_area_classification.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leaf Area Classification GUI with Enhanced Controls and Explanations

This GUI allows users to adjust settings including adaptive thresholding parameters,
kernel size for morphological closing, and logging level. It also features improved
layout and explanations for each parameter.

Features:
---------
- Directory selection with real-time image list loading.
- Configurable processing parameters via GUI inputs with explanations and default values.
- Execution of image processing in a separate thread to keep the GUI responsive.
- Display of processing logs within the application with adjustable log level.
- Image preview functionality on double-clicking image entries.
- Reset to default settings option.
- Save and load configurations seamlessly.
- Stop Processing functionality to terminate ongoing processing tasks.
- Enlarged output panel for better log visibility.
- Improved button layout and appearance.

Author:
-------
Rasmus Jensen
raje at ecos au dk

Date:
-----
October 23, 2024

Version:
--------
0.3.0

"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
import configparser
import sys
import threading
import logging
from typing import List, Dict, Any

# Adjust the system path to include the directory containing leaf_area_classification.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from test_leaf_area_classification import main as process_images_main

# Configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TextHandler(logging.Handler):
    """
    Custom logging handler to display logs in a Tkinter Text widget.

    Attributes:
        text_widget (tk.Text): The Tkinter Text widget where logs will be displayed.
    """

    def __init__(self, text_widget: tk.Text):
        """
        Initialize the TextHandler.

        Args:
            text_widget (tk.Text): The Tkinter Text widget where logs will be displayed.
        """
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record by appending it to the Text widget.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        msg = self.format(record)

        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)

        self.text_widget.after(0, append)


class LeafAreaGUI:
    """
    Leaf Area Classification Graphical User Interface (GUI).

    This class encapsulates the entire GUI application, providing functionalities
    to configure settings, select image directories, execute image processing,
    and display logs and results.
    """

    def __init__(self, root: tk.Tk):
        """
        Initialize the LeafAreaGUI.

        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.root.title("Leaf Area Classification GUI")
        self.root.geometry("1200x900")  # Increased height for larger output panel
        self.root.configure(bg="#f0f0f0")  # Light grey background for a professional look

        # Initialize Config
        self.config_file = os.path.join(current_dir, '..', 'config', 'config.ini')  # Adjusted path
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)

        # Set Font
        self.set_font()

        # Layout frames
        self.create_header()
        self.create_main_frames()

        # Right Frame: Controls
        self.create_controls(self.right_frame)

        # Left Frame: Image File List
        self.create_image_list_panel(self.left_frame)

        # Bottom Frame: Log Output
        self.create_log_panel(self.bottom_frame)

        # Setup logging to the log_text widget
        self.setup_logging()

        # Initialize stop_event
        self.stop_event = threading.Event()

        # Load initial image list if directory is set
        initial_dir = self.config.get("DEFAULT", "image_directory", fallback="")
        if initial_dir and os.path.isdir(initial_dir):
            self.load_image_list(initial_dir)

    def set_font(self) -> None:
        """
        Set the font for the GUI elements.

        Tries to use 'Open Sans' font; falls back to 'Helvetica' if unavailable.
        """
        try:
            self.font_title = ("Open Sans", 24, "bold")
            self.font_subtitle = ("Open Sans", 12, "italic")
            self.font_labels = ("Open Sans", 10)
            self.font_entries = ("Open Sans", 10)
            self.font_buttons = ("Open Sans", 10, "bold")
            self.font_log = ("Open Sans", 10)
        except:
            # Fallback fonts in case Open Sans is not available
            self.font_title = ("Helvetica", 24, "bold")
            self.font_subtitle = ("Helvetica", 12, "italic")
            self.font_labels = ("Helvetica", 10)
            self.font_entries = ("Helvetica", 10)
            self.font_buttons = ("Helvetica", 10, "bold")
            self.font_log = ("Helvetica", 10)

    def create_header(self) -> None:
        """
        Create the header section with title and subtitle.
        """
        header_frame = tk.Frame(self.root, bg="#f0f0f0")
        header_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        # Title
        title_label = tk.Label(header_frame, text="LeafAreaCalc", font=self.font_title, bg="#f0f0f0")
        title_label.pack(anchor='w', padx=20)

        # Subtitle
        subtitle_label = tk.Label(header_frame, text="Comprehensive Leaf Area Calculator",
                                  font=self.font_subtitle, bg="#f0f0f0")
        subtitle_label.pack(anchor='w', padx=20)

    def create_main_frames(self) -> None:
        """
        Create the main frames for controls and image list.
        """
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left Frame: Image List
        self.left_frame = tk.Frame(main_frame, width=1000, bg="#ffffff", bd=2, relief=tk.SUNKEN)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        # Right Frame: Controls
        self.right_frame = tk.Frame(main_frame, bg="#ffffff", bd=2, relief=tk.SUNKEN)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Bottom Frame: Log Output
        self.bottom_frame = tk.Frame(self.root, height=500, bg="#f0f0f0")  # Increased height
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=20, pady=(0, 10))

    def create_controls(self, parent: tk.Frame) -> None:
        """
        Create the controls in the right frame.

        Args:
            parent (tk.Frame): The parent frame where controls will be placed.
        """
        control_frame = tk.Frame(parent, bg="#ffffff")
        control_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Directory Selection
        dir_button = tk.Button(control_frame, text="Select Image Directory", command=self.select_directory,
                               font=self.font_buttons, bg="#2196F3", fg="white", padx=10, pady=5)
        dir_button.grid(row=0, column=0, pady=5, sticky='w', columnspan=3)

        initial_dir = self.config.get("DEFAULT", "image_directory", fallback="No directory selected")
        self.dir_label = tk.Label(control_frame, text=initial_dir, font=self.font_labels, bg="#ffffff")
        self.dir_label.grid(row=1, column=0, pady=5, sticky='w', columnspan=3)

        # Settings Inputs
        settings = [
            ("Area Threshold (mm²)", "area_threshold", "Minimum leaf area to consider in mm². (default 10)"),
            ("Skip Contrast Adjustment", "skip_contrast_adjustment", "Skip CLAHE contrast adjustment. (default False)"),
            ("Image Debug", "img_debug", "Save intermediate images. (default False)"),
            ("Crop Left (%)", "crop_left", "Percentage to crop from the left. (default 20)"),
            ("Crop Right (%)", "crop_right", "Percentage to crop from the right. (default 3)"),
            ("Crop Top (%)", "crop_top", "Percentage to crop from the top. (default 3)"),
            ("Crop Bottom (%)", "crop_bottom", "Percentage to crop from the bottom. (default 3)"),
            ("Filename", "filename", "Specific filename to process. Leave empty to process all images."),
            ("Enable Adaptive Thresholding", "adaptive_threshold", "Enable adaptive thresholding in RGB space. (default True)"),
            ("Adaptive Window Size", "adaptive_window_size", "Window size for adaptive thresholding. Must be an odd integer. (default 15)"),
            ("Adaptive C Constant", "adaptive_C", "Constant subtracted from mean in adaptive thresholding. (default 2)"),
            ("Color Threshold (RGB)", "color_threshold", "RGB value above which pixels are near-white. (default 240)"),
            ("Kernel Size (width,height)", "kernel_size", "Kernel size for morphological closing. Must be a tuple of two integers, e.g., (5,5). (default (5,5))"),
            ("Logging Level", "log_level", "Level of logging detail. (default DEBUG)")
        ]

        self.entries: Dict[str, Any] = {}
        for i, (label_text, key, explanation) in enumerate(settings, start=2):
            label = tk.Label(control_frame, text=label_text + ":", font=self.font_labels, bg="#ffffff")
            label.grid(row=i, column=0, pady=5, sticky='w')

            tooltip = tk.Label(control_frame, text=explanation, font=("Helvetica", 8), fg="grey", bg="#ffffff")
            tooltip.grid(row=i, column=1, pady=5, sticky='w')

            if key in ["skip_contrast_adjustment", "img_debug", "adaptive_threshold"]:
                var = tk.BooleanVar()
                var.set(self.config.getboolean("DEFAULT", key, fallback=False))
                entry = tk.Checkbutton(control_frame, variable=var, bg="#ffffff")
                entry.grid(row=i, column=2, pady=5, sticky='w')
                self.entries[key] = var
            elif key == "log_level":
                options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                var = tk.StringVar()
                var.set(self.config.get("DEFAULT", key, fallback="DEBUG"))
                entry = tk.OptionMenu(control_frame, var, *options)
                entry.config(font=self.font_entries)
                entry.grid(row=i, column=2, pady=5, sticky='w')
                self.entries[key] = var
            else:
                entry = tk.Entry(control_frame, font=self.font_entries)
                entry.grid(row=i, column=2, pady=5, sticky='w')
                default_val = self.config.get("DEFAULT", key, fallback="")
                entry.insert(0, default_val)
                self.entries[key] = entry

        # Create a frame for buttons at the bottom right using grid
        button_frame = tk.Frame(control_frame, bg="#ffffff")
        button_frame.grid(row=len(settings) + 2, column=0, columnspan=3, pady=20, sticky='e')

        # Adjust button style
        button_style = {
            'font': self.font_buttons,
            'bg': "#2196F3",
            'fg': "white",
            'padx': 10,
            'pady': 5,
            'width': 20  # Equal width for all buttons
        }

        # Place the buttons in the button_frame using grid
        save_button = tk.Button(button_frame, text="Save Configuration", command=self.save_config, **button_style)
        save_button.grid(row=0, column=0, padx=5)

        reset_button = tk.Button(button_frame, text="Reset to Defaults", command=self.reset_to_defaults, **button_style)
        reset_button.grid(row=0, column=1, padx=5)

        exec_button = tk.Button(button_frame, text="Execute Processing", command=self.execute_processing, **button_style)
        exec_button.grid(row=0, column=2, padx=5)
        self.exec_button = exec_button  # Reference for enabling/disabling

        stop_button = tk.Button(button_frame, text="Stop Processing", command=self.stop_processing,
                                **button_style, state='disabled')
        stop_button.grid(row=0, column=3, padx=5)
        self.stop_button = stop_button  # Reference for enabling/disabling

    def create_image_list_panel(self, parent: tk.Frame) -> None:
        """
        Create the image file list in the left frame.

        Args:
            parent (tk.Frame): The parent frame where the image list will be placed.
        """
        list_label = tk.Label(parent, text="Image Files:", font=self.font_labels, bg="#ffffff")
        list_label.pack(anchor='w', padx=10, pady=(10, 0))

        self.image_listbox = tk.Listbox(parent, font=self.font_labels)
        self.image_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Bind double-click to display image (optional)
        self.image_listbox.bind('<Double-1>', self.on_image_select)

    def on_image_select(self, event: tk.Event) -> None:
        """
        Handle the event when an image is double-clicked in the list.

        Args:
            event (tk.Event): The event object.
        """
        selection = self.image_listbox.curselection()
        if selection:
            index = selection[0]
            filename = self.image_listbox.get(index)
            directory = self.config.get("DEFAULT", "image_directory", fallback="")
            image_path = os.path.join(directory, filename)
            self.show_image_popup(image_path)

    def show_image_popup(self, image_path: str) -> None:
        """
        Show the selected image in a popup window.

        Args:
            image_path (str): The path to the image file.
        """
        try:
            img = Image.open(image_path)
            img.thumbnail((800, 800))
            popup = tk.Toplevel(self.root)
            popup.title(f"Viewing: {os.path.basename(image_path)}")
            popup.geometry("820x820")
            img_photo = ImageTk.PhotoImage(img)
            img_label = tk.Label(popup, image=img_photo)
            img_label.image = img_photo  # Keep a reference
            img_label.pack()
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to open image:\n{e}")

    def create_log_panel(self, parent: tk.Frame) -> None:
        """
        Create the log output panel in the bottom frame.

        Args:
            parent (tk.Frame): The parent frame where the log panel will be placed.
        """
        log_label = tk.Label(parent, text="Log Output:", font=self.font_labels, bg="#f0f0f0")
        log_label.pack(anchor='w', padx=10, pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(parent, height=20, state='disabled',  # Increased height
                                                  bg="#2e2e2e", fg="#ffffff", font=self.font_log)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def setup_logging(self) -> None:
        """
        Set up logging to redirect to the log_text widget.
        """
        handler = TextHandler(self.log_text)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Set the logging level based on config
        log_level_str = self.config.get("DEFAULT", "log_level", fallback="DEBUG")
        log_level = getattr(logging, log_level_str.upper(), logging.DEBUG)
        logger.setLevel(log_level)

    def select_directory(self) -> None:
        """
        Handle directory selection and load image list.
        """
        directory = filedialog.askdirectory()
        if directory:
            self.dir_label.config(text=directory)
            self.config.set("DEFAULT", "image_directory", directory)
            self.save_config(save_without_message=True)  # Save without showing message
            self.load_image_list(directory)

    def load_image_list(self, directory: str) -> None:
        """
        Load and display the list of image files in the selected directory.

        Args:
            directory (str): The directory containing image files.
        """
        supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        self.images = [f for f in os.listdir(directory) if f.lower().endswith(supported_formats)]

        self.image_listbox.delete(0, tk.END)  # Clear existing list

        if not self.images:
            self.image_listbox.insert(tk.END, "No images found in the selected directory.")
            return

        for img in self.images:
            self.image_listbox.insert(tk.END, img)

    def save_config(self, save_without_message: bool = False) -> None:
        """
        Save the current settings to config.ini.

        Args:
            save_without_message (bool, optional): If True, saves without showing a message.
        """
        directory = self.dir_label.cget("text")
        self.config.set("DEFAULT", "image_directory", directory)
        for key, entry in self.entries.items():
            if key in ["skip_contrast_adjustment", "img_debug", "adaptive_threshold"]:
                self.config.set("DEFAULT", key, str(entry.get()))
            elif key in ["adaptive_window_size", "adaptive_C", "color_threshold"]:
                # Ensure that the entry is an integer
                try:
                    value = int(entry.get())
                    self.config.set("DEFAULT", key, str(value))
                except ValueError:
                    logger.error(f"Invalid value for {key}. It must be an integer.")
                    messagebox.showerror("Invalid Input", f"Invalid value for {key}. It must be an integer.")
                    return
            elif key == "kernel_size":
                # Ensure that the entry is a tuple of integers
                value = entry.get().strip()
                try:
                    # Validate the tuple format
                    if not (value.startswith('(') and value.endswith(')')):
                        raise ValueError
                    kernel_size = tuple(map(int, value[1:-1].split(',')))
                    if len(kernel_size) != 2:
                        raise ValueError
                    self.config.set("DEFAULT", key, str(kernel_size))
                except:
                    logger.error(f"Invalid value for {key}. It must be a tuple of two integers, e.g., (5,5).")
                    messagebox.showerror("Invalid Input", f"Invalid value for {key}. It must be a tuple of two integers, e.g., (5,5).")
                    return
            elif key == "log_level":
                self.config.set("DEFAULT", key, entry.get())
            else:
                self.config.set("DEFAULT", key, entry.get())

        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

        if not save_without_message:
            logger.info("Configuration saved successfully!")
            messagebox.showinfo("Success", "Configuration saved successfully!")

        # Update logging level
        log_level_str = self.config.get("DEFAULT", "log_level", fallback="DEBUG")
        log_level = getattr(logging, log_level_str.upper(), logging.DEBUG)
        logger.setLevel(log_level)

    def reset_to_defaults(self) -> None:
        """
        Reset all settings to default values.
        """
        # Define default values
        defaults = {
            "image_directory": "",
            "area_threshold": "10",
            "skip_contrast_adjustment": "False",
            "img_debug": "False",
            "crop_left": "20",
            "crop_right": "3",
            "crop_top": "3",
            "crop_bottom": "3",
            "filename": "",
            "adaptive_threshold": "True",
            "adaptive_window_size": "15",
            "adaptive_C": "2",
            "color_threshold": "240",
            "kernel_size": "(5, 5)",
            "log_level": "DEBUG"
        }

        # Reset entries
        self.dir_label.config(text=defaults["image_directory"])
        for key, default_val in defaults.items():
            if key in ["skip_contrast_adjustment", "img_debug", "adaptive_threshold"]:
                self.entries[key].set(default_val == "True")
            elif key in ["adaptive_window_size", "adaptive_C", "color_threshold", "kernel_size"]:
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, default_val)
            elif key == "log_level":
                self.entries[key].set(default_val)
            else:
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, default_val)

        # Update config
        for key, default_val in defaults.items():
            self.config.set("DEFAULT", key, default_val)

        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

        # Clear image list
        self.image_listbox.delete(0, tk.END)
        logger.info("Settings reset to default values.")
        messagebox.showinfo("Reset", "Settings have been reset to default values.")

        # Update logging level
        logger.setLevel(logging.DEBUG)

    def execute_processing(self) -> None:
        """
        Execute the image processing in a separate thread.
        """
        # Prevent multiple processing threads
        if self.exec_button['state'] == 'disabled':
            logger.warning("Processing is already running.")
            messagebox.showwarning("Processing", "Processing is already running.")
            return

        # Clear any previous stop event
        self.stop_event.clear()

        # Disable Execute button and enable Stop button
        self.exec_button.config(state='disabled')
        self.stop_button.config(state='normal')

        # Start the processing thread
        thread = threading.Thread(target=self.run_processing, daemon=True)
        thread.start()

    def run_processing(self) -> None:
        """
        Run the image processing script.
        """
        try:
            logger.info("Executing image processing...")
            process_images_main(stop_event=self.stop_event)
            if not self.stop_event.is_set():
                logger.info("Image processing finished.")
                messagebox.showinfo("Processing Complete", "Image processing has completed successfully.")
            else:
                logger.info("Image processing was stopped by the user.")
                messagebox.showinfo("Processing Stopped", "Image processing was stopped.")
            # Optionally, reload image list to reflect any new files
            initial_dir = self.config.get("DEFAULT", "image_directory", fallback="")
            if initial_dir and os.path.isdir(initial_dir):
                self.load_image_list(initial_dir)
        except Exception as e:
            logger.error(f"An error occurred during image processing: {e}")
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
        finally:
            # Re-enable Execute button and disable Stop button
            self.exec_button.config(state='normal')
            self.stop_button.config(state='disabled')

    def stop_processing(self) -> None:
        """
        Signal the processing thread to stop.
        """
        if self.exec_button['state'] == 'normal':
            logger.warning("No processing is currently running.")
            messagebox.showwarning("Stop Processing", "No processing is currently running.")
            return

        self.stop_event.set()
        logger.info("Stop signal sent to processing thread.")
        messagebox.showinfo("Stopping", "Stopping image processing...")

        # Disable Stop button to prevent multiple signals
        self.stop_button.config(state='disabled')


def main() -> None:
    """
    Entry point for the Leaf Area Classification GUI application.
    """
    root = tk.Tk()
    app = LeafAreaGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()