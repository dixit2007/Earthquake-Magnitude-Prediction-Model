import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import threading
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # <--- CHANGED ALGORITHM
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# PART 1: LINEAR REGRESSION ENGINE
# ==========================================
class EarthquakeModel:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.model = None
        self.accuracy_r2 = 0.0
        self.is_trained = False

    def train(self, update_callback=None):
        try:
            # --- STEP 1: LOAD ---
            if update_callback: update_callback("Step 1/4: Loading dataset...")
            df = pd.read_csv(self.csv_file)

            # --- STEP 2: CLEANING ---
            if update_callback: update_callback("Step 2/4: Selecting variables...")
            
            # Select only the numerical physics columns
            vip_columns = ['latitude', 'longitude', 'depth', 'mag']
            df = df[vip_columns].dropna()

            X = df[['latitude', 'longitude', 'depth']]
            y = df['mag']

            # --- STEP 3: SPLIT ---
            if update_callback: update_callback("Step 3/4: Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # --- STEP 4: TRAIN (LINEAR REGRESSION) ---
            if update_callback: update_callback("Step 4/4: Fitting Linear Regression Line...")
            
            # CHANGED: Using Linear Regression instead of Random Forest
            # This draws a straight line through the data points.
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)

            # --- EVALUATE ---
            y_pred = self.model.predict(X_test)
            self.accuracy_r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            if update_callback: update_callback(f"Training Complete! (R² Score: {self.accuracy_r2:.4f})")
            return True, "Success"

        except Exception as e:
            return False, str(e)

    def predict(self, lat, long, depth):
        if not self.model: return None
        input_data = pd.DataFrame([[lat, long, depth]], columns=['latitude', 'longitude', 'depth'])
        return self.model.predict(input_data)[0]

# ==========================================
# PART 2: GUI (SAME SLATE THEME)
# ==========================================
class EarthquakeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Regression Test") # <--- CHANGED TITLE
        self.root.geometry("950x650") 
        self.root.attributes('-alpha', 0.96) 

        # --- THEME COLORS ---
        self.current_theme = "dark"
        self.colors = {
            "dark": {
                "bg": "#0f172a", "card_bg": "#1e293b", "fg": "#f8fafc", "sub_fg": "#94a3b8",
                "accent": "#f43f5e", # Changed accent to Red/Pink for distinction
                "border": "#334155", "entry_bg": "#0f172a", "entry_fg": "#f8fafc",
                "btn_bg": "#f43f5e", "btn_fg": "#ffffff"
            },
            "light": {
                "bg": "#f8fafc", "card_bg": "#ffffff", "fg": "#0f172a", "sub_fg": "#64748b",
                "accent": "#e11d48", "border": "#e2e8f0", "entry_bg": "#f1f5f9", "entry_fg": "#0f172a",
                "btn_bg": "#e11d48", "btn_fg": "#ffffff"
            }
        }

        self.ml_model = EarthquakeModel("earthquake1826_2026.csv")

        self.main_container = tk.Frame(root)
        self.main_container.pack(fill="both", expand=True)

        self.top_bar = tk.Frame(self.main_container, height=70)
        self.top_bar.pack(fill="x", side="top", pady=(0, 20))

        self.content_area = tk.Frame(self.main_container)
        self.content_area.pack(fill="both", expand=True, padx=40, pady=10)

        self.build_ui()
        self.apply_theme()
        self.start_training_thread()

    def build_ui(self):
        # Header
        self.title_label = tk.Label(self.top_bar, text="LINEAR REGRESSION MODEL", font=("Helvetica", 20, "bold"))
        self.title_label.pack(side="left", padx=40, pady=20)

        self.theme_btn = tk.Button(self.top_bar, text="Switch Theme", font=("Arial", 10, "bold"), 
                                   command=self.toggle_theme, bd=0, padx=15, pady=8, cursor="hand2")
        self.theme_btn.pack(side="right", padx=40, pady=20)

        # Inputs
        self.left_card = tk.Frame(self.content_area, bd=1) 
        self.left_card.pack(side="left", fill="both", expand=True, padx=(0, 20))
        
        lbl_input = tk.Label(self.left_card, text="Input Parameters", font=("Helvetica", 14, "bold"))
        lbl_input.pack(anchor="w", pady=(30, 20), padx=30)
        lbl_input.tag = "header"

        self.create_input(self.left_card, "Latitude", "lat_entry", "0.00")
        self.create_input(self.left_card, "Longitude", "long_entry", "0.00")
        self.create_input(self.left_card, "Depth (km)", "depth_entry", "10")

        self.predict_btn = tk.Button(self.left_card, text="RUN LINEAR ANALYSIS", font=("Arial", 11, "bold"), 
                                     state="disabled", cursor="hand2", bd=0, pady=14, command=self.on_predict)
        self.predict_btn.pack(fill="x", padx=30, pady=40)

        # Results
        self.right_card = tk.Frame(self.content_area, bd=1)
        self.right_card.pack(side="right", fill="both", expand=True, padx=(20, 0))

        lbl_mon = tk.Label(self.right_card, text="Live Monitor", font=("Helvetica", 14, "bold"))
        lbl_mon.pack(anchor="w", pady=(30, 10), padx=30)
        lbl_mon.tag = "header"

        self.status_label = tk.Label(self.right_card, text="Initializing Linear Model...", font=("Consolas", 10), anchor="w")
        self.status_label.pack(fill="x", padx=30)

        self.divider = tk.Frame(self.right_card, height=1)
        self.divider.pack(fill="x", padx=30, pady=40)

        lbl_forecast = tk.Label(self.right_card, text="Forecast Magnitude", font=("Helvetica", 11, "bold"))
        lbl_forecast.pack(anchor="c")
        lbl_forecast.tag = "sub"
        
        self.result_val = tk.Label(self.right_card, text="---", font=("Helvetica", 54, "bold"))
        self.result_val.pack(anchor="c", pady=15)
        
        lbl_desc = tk.Label(self.right_card, text="M (Richter)", font=("Arial", 10, "bold"))
        lbl_desc.pack(anchor="c")
        lbl_desc.tag = "sub"

    def create_input(self, parent, label, var_name, placeholder):
        container = tk.Frame(parent)
        container.pack(fill="x", padx=30, pady=10)
        lbl = tk.Label(container, text=label.upper(), font=("Arial", 8, "bold"), anchor="w")
        lbl.pack(fill="x", pady=(0, 5))
        lbl.tag = "sub"
        entry = tk.Entry(container, font=("Arial", 13), bd=0, relief="flat")
        entry.insert(0, placeholder)
        entry.pack(fill="x", ipady=8, padx=10)
        setattr(self, var_name, entry)
        entry.container_ref = container
        entry.lbl_ref = lbl

    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()

    def apply_theme(self):
        c = self.colors[self.current_theme]
        self.root.configure(bg=c["bg"])
        self.main_container.configure(bg=c["bg"])
        self.top_bar.configure(bg=c["bg"])
        self.content_area.configure(bg=c["bg"])
        self.title_label.configure(bg=c["bg"], fg=c["fg"])
        self.theme_btn.configure(bg=c["card_bg"], fg=c["fg"], activebackground=c["accent"], activeforeground="#fff")
        
        for card in [self.left_card, self.right_card]:
            card.configure(bg=c["card_bg"], highlightbackground=c["border"], highlightthickness=1)
            for child in card.winfo_children():
                if isinstance(child, tk.Label):
                    tag = getattr(child, "tag", "")
                    if tag == "header": child.configure(bg=c["card_bg"], fg=c["fg"])
                    elif tag == "sub": child.configure(bg=c["card_bg"], fg=c["sub_fg"])
                    else: child.configure(bg=c["card_bg"], fg=c["fg"])

        for widget in [self.lat_entry, self.long_entry, self.depth_entry]:
            widget.configure(bg=c["entry_bg"], fg=c["entry_fg"], insertbackground=c["fg"])
            widget.container_ref.configure(bg=c["card_bg"])
            widget.lbl_ref.configure(bg=c["card_bg"], fg=c["sub_fg"])

        self.predict_btn.configure(bg=c["btn_bg"], fg=c["btn_fg"], activebackground=c["fg"], activeforeground=c["bg"])
        self.status_label.configure(bg=c["card_bg"], fg=c["accent"])
        self.divider.configure(bg=c["border"])
        self.result_val.configure(bg=c["card_bg"], fg=c["accent"])

    def start_training_thread(self):
        thread = threading.Thread(target=self.run_training)
        thread.daemon = True
        thread.start()

    def run_training(self):
        success, message = self.ml_model.train(self.update_status)
        if success: self.root.after(0, self.on_training_complete)
        else: self.update_status(f"Error: {message}")

    def update_status(self, text):
        self.status_label.config(text=f"> {text}")

    def on_training_complete(self):
        self.predict_btn.config(state="normal")
        self.status_label.config(text=f"System Ready • R²: {self.ml_model.accuracy_r2:.4f}")

    def on_predict(self):
        try:
            lat = float(self.lat_entry.get())
            lng = float(self.long_entry.get())
            depth = float(self.depth_entry.get())
            
            self.status_label.config(text="Processing...")
            prediction = self.ml_model.predict(lat, lng, depth)
            
            self.result_val.config(text=f"{prediction:.2f}")
            self.status_label.config(text="Calculation Complete")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")

if __name__ == "__main__":
    root = tk.Tk()
    app = EarthquakeApp(root)
    root.mainloop()