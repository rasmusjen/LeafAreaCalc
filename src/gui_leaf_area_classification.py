# src/gui_leaf_area_classification.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leaf Area Classification GUI with Enhanced Crop Visualization and RGB Filtering

This GUI allows users to adjust settings including adaptive thresholding parameters,
kernel size for morphological closing, RGB-based filtering options, and logging level.
Additionally, it features an enhanced image preview that displays a red outline indicating
the crop area based on current crop settings.

Features:
--------
- Directory selection with real-time image list loading.
- Configurable processing parameters via GUI inputs with explanations and default values.
- Execution of image processing in a separate thread to keep the GUI responsive.
- Display of processing logs within the application with adjustable log level.
- Image preview functionality on double-clicking image entries, with crop area visualization.
- Reset to default settings option.
- Save and load configurations seamlessly.
- Stop Processing functionality to terminate ongoing processing tasks.
- Enlarged output panel for better log visibility.
- Improved button layout and appearance.
- Additional RGB-based filtering options.
- Option to select and process a single image file.

Author:
-------
Rasmus Jensen
raje [at] ecos [dot] au [dot] dk

Date:
-----
December 3, 2024

Version:
--------
1.2.0

"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import os
import configparser
import sys
import threading
import logging
from typing import List, Dict, Any

# Adjust the system path to include the directory containing leaf_area_classification.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from leaf_area_classification import main as process_images_main  # Updated import


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
        self.root.geometry("1200x800")  # Ensure window size
        self.root.configure(bg="#f0f0f0")  # Light grey background for a professional look

        # Initialize Config
        self.config_file = os.path.join(current_dir, '..', 'config', 'config.ini')  # Adjusted path
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)

        # Set Font
        self.set_font()

        # Initialize entries dictionary
        self.entries: Dict[str, Any] = {}  # Initialize here

        # Layout frames
        self.header_frame = self.create_header()  # Capture header_frame
        self.create_main_frames()

        # Right Frame: Controls
        self.create_controls(self.middle_frame)

        # Left Frame: Image File List
        self.create_image_list_panel(self.left_frame)

        # Bottom Frame: Log Output
        self.create_log_panel(self.right_frame)

        # Setup logging to the log_text widget
        self.setup_logging()

        # Initialize stop_event
        self.stop_event = threading.Event()

        # Load initial image list if directory is set
        initial_dir = self.config.get("DEFAULT", "image_directory", fallback="")
        if initial_dir and os.path.isdir(initial_dir):
            self.load_image_list(initial_dir)
        if initial_dir and os.path.isdir(initial_dir):
            self.load_image_list(initial_dir)
        if initial_dir and os.path.isdir(initial_dir):
            self.load_image_list(initial_dir)

        # Set Application Icon
        try:
            icon_path = os.path.join(current_dir, 'assets', 'LAC.ico')
            self.root.iconbitmap(icon_path)
            # logger.info("Application icon set successfully.")
        except Exception as e:
            logger.error(f"An error occurred during image processing: {e}")
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
            logger.error(f"Failed to set application icon: {e}")

        # Optionally, use the larger icon image within the GUI (e.g., in the header)
        try:
            icon_image_path = os.path.join(current_dir, 'assets', 'icon_new.png')
            icon_image = Image.open(icon_image_path)

            # Resize the image to 10% of its original size
            original_size = icon_image.size
            new_size = (int(original_size[0] * 0.1), int(original_size[1] * 0.1))
            try:
                resample_filter = Image.LANCZOS
            except AttributeError:
                resample_filter = Image.ANTIALIAS  # Fallback for older versions of PIL

            icon_image = icon_image.resize(new_size, resample=resample_filter)

            icon_photo = ImageTk.PhotoImage(icon_image)
            icon_label = tk.Label(self.header_frame, image=icon_photo, bg="#f0f0f0")
            icon_label.image = icon_photo  # Keep a reference to prevent garbage collection

            # Place the icon in the header_frame using grid
            icon_label.grid(row=0, column=1, sticky='e', padx=20)
            # logger.info("Larger icon image added to the header successfully.")
        except Exception as e:
            logger.error(f"Failed to add larger icon image to the header: {e}")

        self.image_popup = None  # Initialize popup reference
        self.current_image_index = None  # Track current image index


    def set_font(self) -> None:
        """
        Set the font for the GUI elements.

        Tries to use 'Open Sans' font; falls back to 'Helvetica' if unavailable.
        """
        try:
            self.font_title = ("Open Sans", 18, "bold")  # Reduced from 24
            self.font_subtitle = ("Open Sans", 10, "italic")  # Reduced from 12
            self.font_labels = ("Open Sans", 8)  # Reduced from 10
            self.font_entries = ("Open Sans", 8)  # Reduced from 10
            self.font_buttons = ("Open Sans", 8, "bold")  # Reduced from 10
            self.font_log = ("Open Sans", 8)  # Reduced from 10
            print("Fonts set successfully using 'Open Sans'.")
            logger.info("Fonts set successfully using 'Open Sans'.")
        except Exception as e:
            # Fallback fonts in case Open Sans is not available
            self.font_title = ("Helvetica", 18, "bold")  # Reduced from 24
            self.font_subtitle = ("Helvetica", 10, "italic")  # Reduced from 12
            self.font_labels = ("Helvetica", 8)  # Reduced from 10
            self.font_entries = ("Helvetica", 8)
            self.font_buttons = ("Helvetica", 8, "bold")
            self.font_log = ("Helvetica", 8)
            print(f"Fonts set using fallback 'Helvetica' due to error: {e}")
            logger.error(f"Fonts set using fallback 'Helvetica' due to error: {e}")


    def create_header(self) -> tk.Frame:
        """
        Create the header section with title and subtitle.
        """
        header_frame = tk.Frame(self.root, bg="#f0f0f0", height=40)  # Set header height to 5% of 800px
        header_frame.pack(side=tk.TOP, fill=tk.X, pady=0)  # Remove vertical padding
        header_frame.pack_propagate(False)  # Prevent frame from resizing based on its content

        # Configure grid with two columns
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=0)

        # Left side: Title and Subtitle in a sub-frame
        title_subtitle_frame = tk.Frame(header_frame, bg="#f0f0f0")
        title_subtitle_frame.grid(row=0, column=0, sticky='w', padx=20)

        # Title
        title_label = tk.Label(title_subtitle_frame, text="LeafAreaCalc", font=self.font_title, bg="#f0f0f0")
        title_label.pack(anchor='w')

        # Subtitle
        subtitle_label = tk.Label(title_subtitle_frame, text="Comprehensive Leaf Area Calculator",
                                font=self.font_subtitle, bg="#f0f0f0")
        subtitle_label.pack(anchor='w')

        # Return header_frame for external access
        return header_frame


    def create_main_frames(self) -> None:
        """
        Create the main frames for image list, controls, and log output.
        """
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left Frame: Image List (15% width)
        self.left_frame = tk.Frame(main_frame, width=210, bg="#ffffff", bd=2, relief=tk.SUNKEN)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), expand=False)
        self.left_frame.pack_propagate(False)  # Prevent frame from resizing

        # Middle Frame: Controls (45% width)
        self.middle_frame = tk.Frame(main_frame, width=510, bg="#ffffff", bd=2, relief=tk.SUNKEN)
        self.middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.middle_frame.pack_propagate(False)  # Prevent frame from resizing

        # Right Frame: Log Output (40% width)
        self.right_frame = tk.Frame(main_frame, width=480, bg="#f0f0f0", bd=2, relief=tk.SUNKEN)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        self.right_frame.pack_propagate(False)  # Prevent frame from resizing


    def create_controls(self, parent: tk.Frame) -> None:
        """
        Create the controls in the middle frame.

        Args:
            parent (tk.Frame): The parent frame where controls will be placed.
        """
        control_frame = tk.Frame(parent, bg="#ffffff")
        control_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Configure grid with 2 columns: 0 (main), 1 (top_right)
        control_frame.grid_columnconfigure(0, weight=1)  # Main settings
        control_frame.grid_columnconfigure(1, weight=0)  # Top right controls

        # Initialize img_debug
        if 'img_debug' not in self.entries:
            self.entries['img_debug'] = tk.BooleanVar()
            self.entries['img_debug'].set(self.config.getboolean("DEFAULT", "img_debug", fallback=False))

        # Initialize log_level
        if 'log_level' not in self.entries:
            self.entries['log_level'] = tk.StringVar()
            self.entries['log_level'].set(self.config.get("DEFAULT", "log_level", fallback="DEBUG"))

        # Directory Selection Frame
        dir_selection_frame = tk.Frame(control_frame, bg="#ffffff")
        dir_selection_frame.grid(row=0, column=0, pady=5, sticky='w')

        # Select Image Directory Button
        dir_button = tk.Button(dir_selection_frame, text="Select Image Directory", command=self.select_directory,
                            font=self.font_buttons, bg="#2196F3", fg="white", padx=5, pady=2)
        dir_button.pack(side=tk.LEFT, padx=(0, 5))

        # Select Single Image Button
        single_image_button = tk.Button(dir_selection_frame, text="Select Single Image", command=self.select_single_image,
                                        font=self.font_buttons, bg="#4CAF50", fg="white", padx=5, pady=2)
        single_image_button.pack(side=tk.LEFT)

        # Top Right Controls Frame
        top_right_frame = tk.Frame(control_frame, bg="#ffffff")
        top_right_frame.grid(row=0, column=1, padx=10, pady=5, sticky='ne')

        # Image Debug Option
        img_debug_var = self.entries.get("img_debug")
        if img_debug_var:
            img_debug_cb = tk.Checkbutton(top_right_frame, text="Image Debug", variable=img_debug_var, bg="#ffffff")
            img_debug_cb.pack(anchor='e')

        # Log Level Dropdown
        log_level_label = tk.Label(top_right_frame, text="Logging Level:", font=self.font_labels, bg="#ffffff")
        log_level_label.pack(anchor='e')
        log_level_menu = self.entries.get("log_level")
        if log_level_menu:
            log_level_menu_widget = tk.OptionMenu(top_right_frame, log_level_menu, "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            log_level_menu_widget.pack(anchor='e')
        initial_dir = self.config.get("DEFAULT", "image_directory", fallback="No directory selected")
        self.dir_label = tk.Label(control_frame, text=initial_dir, font=self.font_labels, bg="#ffffff")
        self.dir_label.grid(row=1, column=0, columnspan=2, pady=5, sticky='w')  # Spanning both columns

        # Image Cropping Section
        cropping_frame = tk.LabelFrame(control_frame, text="Image Cropping", font=self.font_subtitle,
                                    bg="#ffffff", padx=10, pady=10)
        cropping_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky='we')  # Spanning both columns

        # Area Threshold Section
        area_threshold_frame = tk.LabelFrame(control_frame, text="Area Threshold", font=self.font_subtitle,
                                            bg="#ffffff", padx=10, pady=10)
        area_threshold_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky='we')  # Spanning both columns

        area_threshold_settings = [
            ("Area Threshold (mm²):", "area_threshold")
        ]

        for idx, (label_text, key) in enumerate(area_threshold_settings):
            label = tk.Label(area_threshold_frame, text=label_text, font=self.font_labels, bg="#ffffff")
            label.grid(row=idx, column=0, sticky='w', padx=5, pady=2)

            entry = tk.Entry(area_threshold_frame, font=self.font_entries)
            entry.grid(row=idx, column=1, sticky='w', padx=5, pady=2)
            default_val = self.config.get("DEFAULT", key, fallback="2")
            entry.insert(0, default_val)
            self.entries[key] = entry

        # Adaptive Thresholding Section
        adaptive_frame = tk.LabelFrame(control_frame, text="Adaptive Thresholding", font=self.font_subtitle,
                                    bg="#ffffff", padx=10, pady=10)
        adaptive_frame.grid(row=4, column=0, columnspan=2, pady=5, sticky='we')  # Spanning both columns

        adaptive_settings = [
            ("Enable Adaptive Thresholding", "adaptive_threshold"),
            ("Adaptive Window Size", "adaptive_window_size"),
            ("Adaptive C Constant", "adaptive_C"),
            ("Kernel Size (e.g., 3,3)", "kernel_size")  # Added Kernel Size
        ]

        for idx, (label_text, key) in enumerate(adaptive_settings):
            if key == "adaptive_threshold":
                var = self.entries.get(key)
                cb = tk.Checkbutton(adaptive_frame, text=label_text, variable=var, bg="#ffffff",
                                    command=lambda k= key, v=var: self.toggle_adaptive_settings(k, v))
                cb.grid(row=idx, column=0, sticky='w', padx=5, pady=2)
            else:
                label = tk.Label(adaptive_frame, text=label_text + ":", font=self.font_labels, bg="#ffffff")
                label.grid(row=idx, column=0, sticky='w', padx=5, pady=2)

                entry = tk.Entry(adaptive_frame, font=self.font_entries)
                entry.grid(row=idx, column=1, sticky='w', padx=5, pady=2)
                default_val = self.config.get("DEFAULT", key, fallback="")
                entry.insert(0, default_val)
                self.entries[key] = entry

        # Color Threshold Section
        color_threshold_frame = tk.LabelFrame(control_frame, text="Color Threshold", font=self.font_subtitle,
                                            bg="#ffffff", padx=10, pady=10)
        color_threshold_frame.grid(row=5, column=0, columnspan=2, pady=5, sticky='we')  # Spanning both columns

        # Image Cropping Section
        cropping_settings = [
            ("Crop Left (%)", "crop_left"),
            ("Crop Right (%)", "crop_right"),
            ("Crop Top (%)", "crop_top"),
            ("Crop Bottom (%)", "crop_bottom")
        ]

        for idx, (label_text, key) in enumerate(cropping_settings):
            label = tk.Label(cropping_frame, text=label_text + ":", font=self.font_labels, bg="#ffffff")
            label.grid(row=idx, column=0, sticky='w', padx=5, pady=2)

            entry = tk.Entry(cropping_frame, font=self.font_entries)
            entry.grid(row=idx, column=1, sticky='w', padx=5, pady=2)
            default_val = self.config.get("DEFAULT", key, fallback="")
            entry.insert(0, default_val)
            self.entries[key] = entry

        # Adaptive Thresholding Section
        adaptive_settings = [
            ("Enable Adaptive Thresholding", "adaptive_threshold"),
            ("Adaptive Window Size", "adaptive_window_size"),
            ("Adaptive C Constant", "adaptive_C")
        ]

        for idx, (label_text, key) in enumerate(adaptive_settings):
            if key == "adaptive_threshold":
                label = tk.Label(adaptive_frame, text=label_text + ":", font=self.font_labels, bg="#ffffff")
                label.grid(row=idx, column=0, sticky='w', padx=5, pady=2)

                var = self.entries.get(key)
                if not var:
                    var = tk.BooleanVar()
                    var.set(self.config.getboolean("DEFAULT", key, fallback=False))
                    self.entries[key] = var

                cb = tk.Checkbutton(adaptive_frame, variable=var, bg="#ffffff",
                                    command=lambda k=key, v=var: self.toggle_rgb_thresholds(k, v))
                cb.grid(row=idx, column=1, sticky='w', padx=5, pady=2)
            else:
                label = tk.Label(adaptive_frame, text=label_text + ":", font=self.font_labels, bg="#ffffff")
                label.grid(row=idx, column=0, sticky='w', padx=5, pady=2)

                entry = tk.Entry(adaptive_frame, font=self.font_entries)
                entry.grid(row=idx, column=1, sticky='w', padx=5, pady=2)
                default_val = self.config.get("DEFAULT", key, fallback="")
                entry.insert(0, default_val)
                self.entries[key] = entry

        # Color Threshold Section
        color_threshold_settings = [
            ("Color Threshold (RGB):", "color_threshold")
        ]

        for idx, (label_text, key) in enumerate(color_threshold_settings):
            label = tk.Label(color_threshold_frame, text=label_text, font=self.font_labels, bg="#ffffff")
            label.grid(row=idx, column=0, sticky='w', padx=5, pady=2)

            entry = tk.Entry(color_threshold_frame, font=self.font_entries)
            entry.grid(row=idx, column=1, sticky='w', padx=5, pady=2)
            default_val = self.config.get("DEFAULT", key, fallback="")
            entry.insert(0, default_val)
            self.entries[key] = entry

        # Create a frame for action buttons at the bottom of control_frame
        buttons_frame = tk.Frame(control_frame, bg="#ffffff")
        buttons_frame.grid(row=7, column=0, columnspan=2, pady=10, sticky='w')  # Reduced pady

        # Define button style parameters with smaller fonts and increased width
        button_style = {
            'font': self.font_buttons,  # Already reduced
            'bg': "#2196F3",
            'fg': "white",
            'padx': 5,   # Reduced from 10
            'pady': 2,   # Reduced from 5
            'width': 14  # Increased from 10
        }

        # Save Configuration Button
        save_button = tk.Button(buttons_frame, text="Save Configuration", command=self.save_config, **button_style)
        save_button.grid(row=0, column=0, padx=2)  # Reduced padx

        # Reset to Defaults Button
        reset_button = tk.Button(buttons_frame, text="Reset to Defaults", command=self.reset_to_defaults, **button_style)
        reset_button.grid(row=0, column=1, padx=2)  # Reduced padx

        # Execute Processing Button
        exec_button = tk.Button(buttons_frame, text="Execute Processing", command=self.execute_processing, **button_style)
        exec_button.grid(row=0, column=2, padx=2)  # Reduced padx
        self.exec_button = exec_button  # Reference for enabling/disabling

        # Stop Processing Button
        stop_button = tk.Button(buttons_frame, text="Stop Processing", command=self.stop_processing,
                                **button_style, state='disabled')
        stop_button.grid(row=0, column=3, padx=2)  # Reduced padx
        self.stop_button = stop_button  # Reference for enabling/disabling

        # Initially disable RGB threshold entries if filter_rgb is False
        filter_rgb_var = self.entries.get("filter_rgb")
        if filter_rgb_var and not filter_rgb_var.get():
            self.set_rgb_thresholds_state("disabled")

        # Configure grid weights for responsive design
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=0)

        # Reintroduce Kernel Size in Adaptive Thresholding Section
        for idx, (label_text, key) in enumerate(adaptive_settings):
            if key == "kernel_size":
                label = tk.Label(adaptive_frame, text="Kernel Size:", font=self.font_labels, bg="#ffffff")
                label.grid(row=idx, column=0, sticky='w', padx=5, pady=2)

                entry = tk.Entry(adaptive_frame, font=self.font_entries)
                entry.grid(row=idx, column=1, sticky='w', padx=5, pady=2)
                default_val = self.config.get("DEFAULT", key, fallback="(3, 3)")
                entry.insert(0, default_val)
                self.entries[key] = entry

    def reset_to_defaults(self) -> None:
        """
        Reset all settings to default values.
        """
        # Define default values
        defaults = {
            "image_directory": "",
            "area_threshold": "2",
            "adaptive_threshold": "True",
            "adaptive_window_size": "15",
            "adaptive_C": "2",
            "color_threshold": "180",
            "kernel_size": "(3, 3)",
            "log_level": "DEBUG"
        }

        # Reset entries
        self.dir_label.config(text=defaults["image_directory"])
        for key, default_val in defaults.items():
            if key in ["adaptive_threshold"]:
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

        # Disable RGB threshold entries if filter_rgb is False
        filter_rgb_var = self.entries.get("filter_rgb")
        if filter_rgb_var and not filter_rgb_var.get():
            self.set_rgb_thresholds_state("disabled")
        else:
            self.set_rgb_thresholds_state("normal")


    def toggle_rgb_thresholds(self, key: str, var: tk.BooleanVar) -> None:
        """
        Enable or disable RGB threshold entries based on the RGB filter checkbox.

        Args:
            key (str): The key corresponding to the RGB filter.
            var (tk.BooleanVar): The BooleanVar linked to the RGB filter checkbox.
        """
        state = "normal" if var.get() else "disabled"
        self.set_rgb_thresholds_state(state)

    def set_rgb_thresholds_state(self, state: str) -> None:
        """
        Set the state of RGB threshold entry fields.

        Args:
            state (str): The state to set ('normal' or 'disabled').
        """
        for key in ["r_threshold", "g_threshold", "b_threshold"]:
            widget = self.entries.get(key)
            if isinstance(widget, tk.Entry):
                widget.config(state=state)

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
            self.show_image_popup(image_path, current_index=index)

    def show_image_popup(self, image_path: str, current_index: int) -> None:
        """
        Show the selected image in a popup window with a red outline indicating the crop area.
        If the popup already exists, update the image instead of creating a new window.

        Args:
            image_path (str): The path to the image file.
            current_index (int): The index of the current image.
        """
        try:
            img = Image.open(image_path)
            img_copy = img.copy()  # Make a copy to draw on
            draw = ImageDraw.Draw(img_copy)

            # Retrieve crop settings from GUI
            crop_left = self.get_crop_percentage("crop_left")
            crop_right = self.get_crop_percentage("crop_right")
            crop_top = self.get_crop_percentage("crop_top")
            crop_bottom = self.get_crop_percentage("crop_bottom")

            # Calculate crop coordinates based on percentages
            img_width, img_height = img_copy.size
            left_px = int((crop_left / 100) * img_width)
            right_px = int(img_width - (crop_right / 100) * img_width)
            top_px = int((crop_top / 100) * img_height)
            bottom_px = int(img_height - (crop_bottom / 100) * img_height)

            # Draw red rectangle outlining the crop area
            draw.rectangle([(left_px, top_px), (right_px, bottom_px)], outline="red", width=3)

            # Resize image for display if it's too large
            max_size = (600, 400)  # Reduced from (800, 800)
            try:
                resample_filter = Image.ANTIALIAS
            except AttributeError:
                resample_filter = Image.LANCZOS

            img_copy.thumbnail(max_size, resample=resample_filter)
            resized_width, resized_height = img_copy.size

            img_photo = ImageTk.PhotoImage(img_copy)

            if self.image_popup and self.image_popup.winfo_exists():
                # ...existing code...
                self.image_popup.image_label.config(image=img_photo)
                self.image_popup.image_label.image = img_photo  # Prevent garbage collection
                # ...existing code...
            else:
                # Create a new popup window
                self.image_popup = tk.Toplevel(self.root)
                self.image_popup.title("Image Preview")

                # Calculate additional height for buttons (10% of image height)
                additional_height = int(resized_height * 0.1)
                popup_height = resized_height + additional_height
                popup_width = resized_width

                self.image_popup.geometry(f"{popup_width}x{popup_height}")  # Set dynamic popup window size
                self.image_popup.resizable(False, False)  # Prevent resizing

                # Display the image in the popup
                self.image_popup.image_label = tk.Label(self.image_popup, image=img_photo)
                self.image_popup.image_label.image = img_photo  # Keep a reference
                self.image_popup.image_label.pack()

                # Add navigation buttons below the image
                nav_frame = tk.Frame(self.image_popup, bg="#f0f0f0")
                nav_frame.pack(pady=5)  # Add padding to separate buttons from the image

                prev_button = tk.Button(nav_frame, text="Previous", command=self.show_previous_image, width=10)
                prev_button.pack(side=tk.LEFT, padx=10)

                next_button = tk.Button(nav_frame, text="Next", command=self.show_next_image, width=10)
                next_button.pack(side=tk.LEFT, padx=10)

            self.current_image_index = current_index  # Update current index
            logger.info(f"Displayed image with crop outline: {os.path.basename(image_path)}")

        except Exception as e:
            logger.error(f"Failed to open image: {image_path}. Error: {e}")
            messagebox.showerror("Image Error", f"Failed to open image:\n{e}")

    def show_previous_image(self) -> None:
        """
        Display the previous image in the popup window.
        """
        if self.current_image_index is not None and self.current_image_index > 0:
            self.current_image_index -= 1
            image_path = self.get_image_path(self.current_image_index)
            self.show_image_popup(image_path, self.current_image_index)

    def show_next_image(self) -> None:
        """
        Display the next image in the popup window.
        """
        if self.current_image_index is not None and self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            image_path = self.get_image_path(self.current_image_index)
            self.show_image_popup(image_path, self.current_image_index)

    def get_crop_percentage(self, key: str) -> float:
        """
        Retrieve and validate crop percentage from the GUI entry.

        Args:
            key (str): The key corresponding to the crop setting.

        Returns:
            float: The validated crop percentage.

        Raises:
            ValueError: If the entry is not a valid float between 0 and 100.
        """
        try:
            entry_widget = self.entries[key]
            value = float(entry_widget.get())
            if not (0 <= value <= 100):
                raise ValueError
            return value
        except ValueError:
            logger.error(f"Invalid value for {key}. Must be a number between 0 and 100.")
            messagebox.showerror("Invalid Input", f"Invalid value for {key.replace('_', ' ').title()}. Must be a number between 0 and 100.")
            return 0.0  # Default fallback

    def create_log_panel(self, parent: tk.Frame) -> None:
        """
        Create the log output panel in the right frame.

        Args:
            parent (tk.Frame): The parent frame where the log panel will be placed.
        """
        log_label = tk.Label(parent, text="Log Output:", font=self.font_labels, bg="#f0f0f0")
        log_label.pack(anchor='w', padx=10, pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(parent, height=30, state='disabled',  # Adjust height for 30% width
                                                  bg="#2e2e2e", fg="#ffffff", font=self.font_log)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def setup_logging(self) -> None:
        """
        Set up logging to redirect to the log_text widget.
        """
        handler = TextHandler(self.log_text)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root_logger = logging.getLogger()  # Get the root logger
        root_logger.addHandler(handler)
        # Set the logging level based on config
        log_level_str = self.config.get("DEFAULT", "log_level", fallback="DEBUG")
        log_level = getattr(logging, log_level_str.upper(), logging.DEBUG)
        root_logger.setLevel(log_level)

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

    def select_single_image(self) -> None:
        """
        Handle single image selection and add it to the image list.
        """
        filetypes = [
            ("Image Files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
            ("All Files", "*.*")
        ]
        image_path = filedialog.askopenfilename(title="Select Single Image", filetypes=filetypes)
        if image_path:
            directory = os.path.dirname(image_path)
            filename = os.path.basename(image_path)
            # Update directory label to the directory of the selected image
            self.dir_label.config(text=directory)
            self.config.set("DEFAULT", "image_directory", directory)
            self.save_config(save_without_message=True)  # Save without showing message
            # Load image list from the directory
            self.load_image_list(directory)
            # Add the selected single image to the listbox if it's not already present
            if filename not in self.images:
                self.image_listbox.insert(tk.END, filename)
                self.images.append(filename)

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
            if key in ["adaptive_threshold", "img_debug"]:  # Removed "filter_rgb" here
                self.config.set("DEFAULT", key, str(entry.get()))
            elif key in ["adaptive_window_size", "adaptive_C", "color_threshold"]:
                # Ensure that the entry is an integer
                try:
                    value = int(entry.get())
                    self.config.set("DEFAULT", key, str(value))
                except ValueError:
                    logger.error(f"Invalid value for {key}. It must be an integer.")
                    messagebox.showerror("Invalid Input", f"Invalid value for {key.replace('_', ' ').title()}. It must be an integer.")
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
                    logger.error(f"Invalid value for {key}. It must be a tuple of two integers, e.g., (3,3).")
                    messagebox.showerror("Invalid Input", f"Invalid value for {key.replace('_', ' ').title()}. It must be a tuple of two integers, e.g., (3,3).")
                    return
            elif key == "log_level":
                self.config.set("DEFAULT", key, entry.get())
            else:
                # For any other keys, ensure they are strings
                self.config.set("DEFAULT", key, str(entry.get()))  # Changed to str(entry.get())

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
                "area_threshold": "2",
                "adaptive_threshold": "True",
                "adaptive_window_size": "15",
                "adaptive_C": "2",
                "color_threshold": "180",
                "kernel_size": "(3, 3)",
                "log_level": "DEBUG"
            }

            # Reset entries
            self.dir_label.config(text=defaults["image_directory"])
            for key, default_val in defaults.items():
                if key in ["adaptive_threshold"]:
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

        # Disable RGB threshold entries if filter_rgb is False
        filter_rgb_var = self.entries.get("filter_rgb")
        if filter_rgb_var and not filter_rgb_var.get():
            self.set_rgb_thresholds_state("disabled")
        else:
            self.set_rgb_thresholds_state("normal")

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
            result_dir = process_images_main(stop_event=self.stop_event)  # Capture result directory
            if not self.stop_event.is_set():
                # Processing completed successfully
                logger.info("Processing completed successfully.")

                # Prompt to open the result directory
                if result_dir and os.path.isdir(result_dir):
                    open_dir = messagebox.askyesno(
                        "Processing Complete",
                        f"Image processing completed successfully.\nDo you want to open the result directory?\n{result_dir}"
                    )
                    if open_dir:
                        try:
                            os.startfile(result_dir)
                        except Exception as e:
                            logger.error(f"Failed to open directory: {result_dir}. Error: {e}")
                            messagebox.showerror("Directory Error", f"Failed to open directory:\n{e}")
                
            else:
                # Processing was stopped
                logger.info("Processing was stopped by the user.")
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

    def select_single_image(self) -> None:
        """
        Handle single image selection and add it to the image list.
        """
        filetypes = [
            ("Image Files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
            ("All Files", "*.*")
        ]
        image_path = filedialog.askopenfilename(title="Select Single Image", filetypes=filetypes)
        if image_path:
            directory = os.path.dirname(image_path)
            filename = os.path.basename(image_path)
            # Update directory label to the directory of the selected image
            self.dir_label.config(text=directory)
            self.config.set("DEFAULT", "image_directory", directory)
            self.save_config(save_without_message=True)  # Save without showing message
            # Load image list from the directory
            self.load_image_list(directory)
            # Add the selected single image to the listbox if it's not already present
            if filename not in self.images:
                self.image_listbox.insert(tk.END, filename)
                self.images.append(filename)
    
    def update_status(self, message: str) -> None:
        """
        Update the status bar with the given message.

        Args:
            message (str): The message to display in the status bar.
        """
        self.status_label.after(0, lambda: self.status_label.config(text=message))

    def get_image_path(self, index: int) -> str:
        """
        Retrieve the file path of the image at the specified index.

        Args:
            index (int): The index of the image in the images list.

        Returns:
            str: The full file path of the image.
        """
        directory = self.config.get("DEFAULT", "image_directory", fallback="")
        filename = self.images[index]
        return os.path.join(directory, filename)


def main() -> None:
    """
    Entry point for the Leaf Area Classification GUI application.
    """
    root = tk.Tk()
    app = LeafAreaGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
