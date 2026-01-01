# Loan Approval ML Predictor

A full-stack application for predicting loan approval using Machine Learning models (Logistic Regression and Decision Tree).

## Project Structure

```
loan_approval/
├── api/
│   ├── predict.py          # Serverless function for ML predictions
│   ├── requirements.txt    # Python dependencies
│   └── ml_models/          # ML model files (.pkl)
├── src/                    # React frontend source code
├── public/                 # React public assets
├── package.json            # Frontend dependencies
├── vercel.json            # Vercel deployment configuration
└── .vercelignore          # Files to ignore during deployment
```

## Features

- **Frontend**: React application with a user-friendly form for loan application data
- **Backend**: Python serverless function with scikit-learn ML models
- **Models**: 
  - Logistic Regression
  - Decision Tree

## Local Development

### Frontend
```bash
npm install
npm start
```

### Backend (if testing locally)
The backend is designed to run as a Vercel serverless function. For local testing, you can use the original Django backend in the parent directory.

## Deployment on Vercel

1. **Push to GitHub**
   - Create a new repository
   - Push only the `loan_approval` folder contents

2. **Deploy on Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New Project"
   - Import your GitHub repository
   - Vercel will auto-detect the configuration
   - Click "Deploy"

3. **That's it!** 
   - The React frontend will be served as a static site
   - The `/api/predict` endpoint will run as a Python serverless function

## API Endpoint

**POST /api/predict**

Request body:
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 2000,
  "LoanAmount": 100000,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

Response:
```json
{
  "logistic_regression": "Approved",
  "decision_tree": "Approved"
}
```

## Notes

- The ML models are loaded on cold start and cached for performance
- The frontend uses relative API paths, so it works automatically on Vercel
- Python dependencies are automatically installed from `api/requirements.txt`
- Optimized for Vercel's serverless function size limits (removed pandas dependency)

