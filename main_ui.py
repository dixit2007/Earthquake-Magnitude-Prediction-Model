import tkinter as tk
from tkinter import messagebox
import threading
from model_logic import EarthquakePredictor  # Import the logic from the other file

class AppWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("SeismicAI System")
        self.root.geometry("900x600")
        self.root.configure(bg="#0f172a") # Dark Blue Background

        # Connect to the ML Logic File
        self.brain = EarthquakePredictor("earthquake1826_2026.csv")
        self.input_boxes = {} # To store the 8 text boxes

        self.setup_screen()
        
        # Start training in background so window doesn't freeze
        threading.Thread(target=self.start_backend_training, daemon=True).start()

    def setup_screen(self):
        # 1. Heading
        tk.Label(self.root, text="EARTHQUAKE PREDICTOR", font=("Arial", 20, "bold"), 
                 bg="#0f172a", fg="white").pack(pady=20)

        # 2. Main Layout (Left for Inputs, Right for Result)
        main_frame = tk.Frame(self.root, bg="#0f172a")
        main_frame.pack(fill="both", expand=True, padx=30)

        left_side = tk.Frame(main_frame, bg="#1e293b", padx=20, pady=20)
        left_side.pack(side="left", fill="both", expand=True, padx=10)

        right_side = tk.Frame(main_frame, bg="#1e293b", padx=20, pady=20)
        right_side.pack(side="right", fill="both", expand=True, padx=10)

        # 3. Create 8 Input Fields (Left Side)
        # List of: (Label Name, Internal ID, Default Value)
        fields = [
            ("Latitude", "latitude", "35.0"),
            ("Longitude", "longitude", "-118.0"),
            ("Depth (km)", "depth", "10.0"),
            ("Stations (NST)", "nst", "25"),
            ("Gap Angle", "gap", "50"),
            ("Min Distance", "dmin", "0.5"),
            ("RMS Error", "rms", "0.2"),
            ("Loc Error", "horizontalError", "1.5")
        ]

        tk.Label(left_side, text="INPUT DATA", font=("Arial", 12, "bold"), bg="#1e293b", fg="#38bdf8").pack(anchor="w", pady=10)

        for label, key, default in fields:
            self.create_entry(left_side, label, key, default)

        self.btn_predict = tk.Button(left_side, text="PREDICT MAGNITUDE", bg="#38bdf8", font=("Arial", 10, "bold"),
                                     state="disabled", command=self.calculate_result)
        self.btn_predict.pack(fill="x", pady=20)

        # 4. Result Area (Right Side)
        tk.Label(right_side, text="SYSTEM STATUS", font=("Arial", 12, "bold"), bg="#1e293b", fg="#38bdf8").pack(anchor="w", pady=10)
        
        self.lbl_status = tk.Label(right_side, text="Initializing...", fg="yellow", bg="#1e293b", font=("Consolas", 10))
        self.lbl_status.pack(anchor="w")

        tk.Label(right_side, text="MAGNITUDE", fg="#94a3b8", bg="#1e293b").pack(pady=(60,5))
        
        self.lbl_result = tk.Label(right_side, text="--", font=("Arial", 60, "bold"), fg="#38bdf8", bg="#1e293b")
        self.lbl_result.pack()

    def create_entry(self, parent, text, key, default):
        """Helper to make text boxes neatly"""
        row = tk.Frame(parent, bg="#1e293b")
        row.pack(fill="x", pady=2)
        tk.Label(row, text=text, width=15, anchor="w", bg="#1e293b", fg="white").pack(side="left")
        entry = tk.Entry(row, bg="#0f172a", fg="white", insertbackground="white")
        entry.insert(0, default)
        entry.pack(side="right", fill="x", expand=True)
        self.input_boxes[key] = entry

    def start_backend_training(self):
        """Runs automatically when app opens"""
        success = self.brain.train_model(self.update_status_text)
        if success:
            self.btn_predict.config(state="normal") # Enable button when done

    def update_status_text(self, text):
        self.lbl_status.config(text=f"> {text}")

    def calculate_result(self):
        try:
            # Gather all 8 numbers from the boxes
            values = []
            for key in self.brain.features:
                val = float(self.input_boxes[key].get())
                values.append(val)

            # Send to model
            self.update_status_text("Calculating...")
            prediction = self.brain.get_prediction(values)
            
            # Show result
            self.lbl_result.config(text=f"{prediction:.2f}")
            self.update_status_text("Done.")

        except ValueError:
            messagebox.showerror("Error", "Please enter numbers only.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppWindow(root)
    root.mainloop()