import React, { useState } from "react";

function App() {
  const [formData, setFormData] = useState({
    Gender: "Male",
    Married: "Yes",
    Dependents: "0",
    Education: "Graduate",
    Self_Employed: "No",
    ApplicantIncome: "",
    CoapplicantIncome: "",
    LoanAmount: "",
    Loan_Amount_Term: 360,
    Credit_History: 1,
    Property_Area: "Urban"
  });

  const [prediction, setPrediction] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const predict = async (modelType) => {
    setSelectedModel(modelType);
    setLoading(true);
    setPrediction(null);

    // Use environment variable for API URL, fallback to relative path for Vercel
    const apiUrl = process.env.REACT_APP_API_URL || "/api/predict";
    
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...formData,
        ApplicantIncome: Number(formData.ApplicantIncome),
        CoapplicantIncome: Number(formData.CoapplicantIncome),
        LoanAmount: Number(formData.LoanAmount),
        Loan_Amount_Term: Number(formData.Loan_Amount_Term),
        Credit_History: Number(formData.Credit_History)
      })
    });

    const data = await response.json();

    setPrediction(
      modelType === "logistic"
        ? data.logistic_regression
        : data.decision_tree
    );

    setLoading(false);
  };

  return (
    <div className="container">
      <div className="card">
        <h1 className="title">Loan Approval Predictor</h1>
        <p className="subtitle">
          Predict loan approval using Machine Learning
        </p>

        <div className="form">
          {Object.keys(formData).map((key) => (
            <div className="input-group" key={key}>
              <label>{key}</label>
              <input
                name={key}
                value={formData[key]}
                onChange={handleChange}
              />
            </div>
          ))}
        </div>

        <div className="buttons">
          <button className="btn blue" onClick={() => predict("logistic")}>
            Logistic Regression
          </button>

          <button className="btn green" onClick={() => predict("tree")}>
            Decision Tree
          </button>
        </div>

        {loading && <p className="loading">Predicting...</p>}

        {prediction && (
          <div className="result">
            <h3>Prediction Result</h3>
            <p>
              <b>
                {selectedModel === "logistic"
                  ? "Logistic Regression"
                  : "Decision Tree"}
                :
              </b>{" "}
              {prediction}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
