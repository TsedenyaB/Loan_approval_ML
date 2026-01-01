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

    try {
      // Use environment variable for API URL, fallback to relative path for Vercel
      const apiUrl = process.env.REACT_APP_API_URL || "/api/predict";
      
      const requestData = {
        ...formData,
        ApplicantIncome: Number(formData.ApplicantIncome) || 0,
        CoapplicantIncome: Number(formData.CoapplicantIncome) || 0,
        LoanAmount: Number(formData.LoanAmount) || 0,
        Loan_Amount_Term: Number(formData.Loan_Amount_Term) || 360,
        Credit_History: Number(formData.Credit_History) || 1
      };

      console.log("Sending request to:", apiUrl);
      console.log("Request data:", requestData);

      // Add timeout to fetch request
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      console.log("Response status:", response.status);
      console.log("Response ok:", response.ok);

      const responseText = await response.text();
      console.log("Response text:", responseText);

      if (!response.ok) {
        let errorData;
        try {
          errorData = JSON.parse(responseText);
        } catch {
          errorData = { error: responseText || `HTTP error! status: ${response.status}` };
        }
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      let data;
      try {
        data = JSON.parse(responseText);
      } catch (e) {
        throw new Error(`Invalid JSON response: ${responseText}`);
      }

      console.log("Parsed data:", data);

      if (data.error) {
        throw new Error(data.error);
      }

      if (!data.logistic_regression || !data.decision_tree) {
        throw new Error("Invalid response format from API");
      }

      setPrediction(
        modelType === "logistic"
          ? data.logistic_regression
          : data.decision_tree
      );
    } catch (error) {
      console.error("Prediction error:", error);
      if (error.name === 'AbortError') {
        setPrediction("Error: Request timed out. Please try again.");
      } else {
        setPrediction(`Error: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
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
