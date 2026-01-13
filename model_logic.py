import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class EarthquakePredictor:
    def __init__(self, filename):
        self.filename = filename
        self.model = None
        self.accuracy = 0.0
        # The 8 technical columns we need
        self.features = ['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms', 'horizontalError']

    def train_model(self, status_callback):
        """
        Loads data, cleans it, and trains the AI.
        """
        try:
            status_callback("Loading dataset...")
            df = pd.read_csv(self.filename)

            # 1. Fill missing technical data with 0 or average to prevent errors
            # (Old earthquakes often miss 'gap' or 'nst' data)
            missing_cols = [col for col in self.features if col not in df.columns]
            for col in missing_cols:
                df[col] = 0.0
            
            df = df.fillna(0) # Fill any remaining empty cells with 0

            # 2. Separate Inputs (X) and Output (y)
            X = df[self.features]
            y = df['mag']

            # 3. Split: 80% to Study, 20% to Test
            status_callback("Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 4. Train the Random Forest
            status_callback("Training brain (100 Trees)...")
            self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            self.model.fit(X_train, y_train)

            # 5. Check Accuracy
            prediction_test = self.model.predict(X_test)
            self.accuracy = r2_score(y_test, prediction_test)

            status_callback(f"Ready! Accuracy: {self.accuracy:.2f}")
            return True

        except Exception as error_msg:
            status_callback(f"Error: {str(error_msg)}")
            return False

    def get_prediction(self, inputs_list):
        """
        Takes a list of 8 numbers and returns the predicted magnitude.
        """
        if not self.model: return 0.0
        
        # Convert list to DataFrame format
        data_row = pd.DataFrame([inputs_list], columns=self.features)
        result = self.model.predict(data_row)
        return result[0]