from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import joblib
import os

# Load models (cached on cold start)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lr_model = None
dt_model = None
feature_names = None

def load_models():
    global lr_model, dt_model, feature_names
    if lr_model is None:
        lr_model = joblib.load(os.path.join(BASE_DIR, "ml_models/logistic_regression_model.pkl"))
    if dt_model is None:
        dt_model = joblib.load(os.path.join(BASE_DIR, "ml_models/decision_tree_model.pkl"))
    
    # Try to get feature names from model (sklearn 1.0+)
    if feature_names is None:
        if hasattr(lr_model, 'feature_names_in_'):
            feature_names = list(lr_model.feature_names_in_)
        elif hasattr(dt_model, 'feature_names_in_'):
            feature_names = list(dt_model.feature_names_in_)
        else:
            # Default feature order (typical loan dataset)
            feature_names = [
                'Gender', 'Married', 'Dependents', 'Education', 
                'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
            ]

def convert_to_array(data):
    """
    Convert dictionary to numpy array in the order expected by the model.
    """
    global feature_names
    values = []
    for feature in feature_names:
        value = data.get(feature, 0)
        
        # Convert categorical to numeric if needed
        if feature == 'Gender':
            value = 1 if str(value).lower() in ['male', '1', 'yes'] else 0
        elif feature == 'Married':
            value = 1 if str(value).lower() in ['yes', '1', 'married'] else 0
        elif feature == 'Education':
            value = 1 if str(value).lower() in ['graduate', '1', 'yes'] else 0
        elif feature == 'Self_Employed':
            value = 1 if str(value).lower() in ['yes', '1'] else 0
        elif feature == 'Property_Area':
            # Urban=2, Semiurban=1, Rural=0 (typical encoding)
            prop = str(value).lower()
            value = 2 if 'urban' in prop else (1 if 'semi' in prop else 0)
        else:
            # Convert to float for numeric features
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0.0
        
        values.append(value)
    
    return np.array([values], dtype=np.float64)

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        try:
            # Load models
            load_models()
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            # Convert to numpy array
            features = convert_to_array(data)
            
            # Make predictions
            lr_pred = lr_model.predict(features)[0]
            dt_pred = dt_model.predict(features)[0]
            
            # Prepare response
            response = {
                "logistic_regression": "Approved" if lr_pred == 1 else "Rejected",
                "decision_tree": "Approved" if dt_pred == 1 else "Rejected"
            }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            # Error response
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_response = {"error": str(e)}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

