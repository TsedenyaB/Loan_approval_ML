from http.server import BaseHTTPRequestHandler
import json
import pandas as pd
import joblib
import os

# Load models (cached on cold start)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lr_model = None
dt_model = None

def load_models():
    global lr_model, dt_model
    if lr_model is None:
        lr_model = joblib.load(os.path.join(BASE_DIR, "ml_models/logistic_regression_model.pkl"))
    if dt_model is None:
        dt_model = joblib.load(os.path.join(BASE_DIR, "ml_models/decision_tree_model.pkl"))

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
            
            # Convert to DataFrame (exactly as the original code did)
            df = pd.DataFrame([data])
            
            # Make predictions
            lr_pred = lr_model.predict(df)[0]
            dt_pred = dt_model.predict(df)[0]
            
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

