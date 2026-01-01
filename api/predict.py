from http.server import BaseHTTPRequestHandler
import json
import pandas as pd
import joblib
import os
import warnings

# Suppress warnings but allow model loading
import sys
warnings.filterwarnings('ignore', category=UserWarning)
# Don't suppress the version warning - we need to see it

# Load models (cached on cold start)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lr_model = None
dt_model = None

def load_models():
    global lr_model, dt_model
    try:
        # Check sklearn version - MUST be 1.6.1 for model compatibility
        import sklearn
        sklearn_version = sklearn.__version__
        print(f"Loaded scikit-learn version: {sklearn_version}")
        
        # Models were trained with 1.6.1 - enforce this version
        if sklearn_version != "1.6.1":
            raise Exception(
                f"Version mismatch! Models require scikit-learn 1.6.1, "
                f"but {sklearn_version} is installed. Please update requirements.txt"
            )
        
        if lr_model is None:
            model_path = os.path.join(BASE_DIR, "ml_models", "logistic_regression_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            # Load with warnings suppressed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr_model = joblib.load(model_path)
            print("Logistic Regression model loaded successfully")
        
        if dt_model is None:
            model_path = os.path.join(BASE_DIR, "ml_models", "decision_tree_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            # Load with warnings suppressed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt_model = joblib.load(model_path)
            print("Decision Tree model loaded successfully")
    except Exception as e:
        import traceback
        error_msg = f"Error loading models: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise Exception(error_msg)

class handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default logging
        pass
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        try:
            # Load models first
            load_models()
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                raise ValueError("Empty request body")
            
            body = self.rfile.read(content_length)
            if not body:
                raise ValueError("Could not read request body")
            
            # Parse JSON
            try:
                data = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {str(e)}")
            
            # Ensure all required fields are present and convert types
            required_fields = [
                'Gender', 'Married', 'Dependents', 'Education', 
                'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
            ]
            
            # Prepare data with proper types
            processed_data = {}
            for field in required_fields:
                value = data.get(field, None)
                if value is None:
                    raise ValueError(f"Missing required field: {field}")
                
                # Convert numeric fields
                if field in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
                    try:
                        processed_data[field] = float(value) if value != '' else 0.0
                    except (ValueError, TypeError):
                        processed_data[field] = 0.0
                elif field == 'Dependents':
                    try:
                        processed_data[field] = str(value).strip() if value != '' else '0'
                    except:
                        processed_data[field] = '0'
                else:
                    processed_data[field] = str(value).strip() if value != '' else ''
            
            # Convert to DataFrame (exactly as the original code did)
            df = pd.DataFrame([processed_data])
            
            # Make predictions
            try:
                lr_pred = lr_model.predict(df)[0]
                dt_pred = dt_model.predict(df)[0]
            except Exception as e:
                raise Exception(f"Prediction error: {str(e)}")
            
            # Convert predictions to int if needed
            lr_pred = int(lr_pred) if hasattr(lr_pred, '__int__') else lr_pred
            dt_pred = int(dt_pred) if hasattr(dt_pred, '__int__') else dt_pred
            
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
            import traceback
            error_trace = traceback.format_exc()
            # Error response - keep it simple for Vercel
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Log the full error (Vercel will capture this)
            print(f"ERROR: {error_type}: {error_msg}")
            print(f"TRACEBACK: {error_trace}")
            
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            # Send simplified error response
            error_response = {
                "error": error_msg,
                "type": error_type
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
