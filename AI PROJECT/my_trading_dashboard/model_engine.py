import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class ModelEngine:
    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.model_metrics = {}
        self.best_model_name = None
        self.best_model = None

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Trains all models and evaluates them on the test set.
        Returns a dictionary of metrics for each model.
        """
        results = {}
        best_r2 = -float('inf')
        
        for name, model in self.models.items():
            # Train
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Predict
            predictions = model.predict(X_test)
            
            # Evaluate
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            results[name] = {
                "R2": round(r2, 4), 
                "MAE": round(mae, 4), 
                "RMSE": round(rmse, 4)
            }
            
            # Selection logic (track best based on R2)
            if r2 > best_r2:
                best_r2 = r2
                self.best_model_name = name
                self.best_model = model
                
        self.model_metrics = results
        return results

    def get_best_model_info(self):
        """Returns the best model instance, name, and its metrics."""
        if not self.best_model:
            return None, None, None
        return self.best_model, self.best_model_name, self.model_metrics[self.best_model_name]

    def forecast_future(self, model, last_window, steps=5):
        """
        Generates future predictions using recursive forecasting.
        
        Args:
            model: Trained model instance
            last_window: The last sequence of data points (shape: [steps, 1] or [steps,])
            steps: Number of days to predict
            
        Returns:
            List of predicted values
        """
        predictions = []
        # Ensure window is in correct shape for prediction (1, window_size)
        current_window = np.array(last_window).flatten()
        
        for _ in range(steps):
            # Reshape for sklearn model (1, features)
            input_seq = current_window.reshape(1, -1)
            
            # Predict next step
            pred = model.predict(input_seq)[0]
            predictions.append(pred)
            
            # Update window: remove oldest, add new prediction
            current_window = np.append(current_window[1:], pred)
            
        return predictions
