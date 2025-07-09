from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import traceback

app = Flask(__name__)

# Print current working directory (for debugging)
print("Current Directory:", os.getcwd())

# === Load Preprocessing Objects and Model ===
scaler = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')
best_model = joblib.load('best_model.pkl')
to_drop = joblib.load('to_drop.pkl')

# === Expected Feature Columns ===
feature_columns = [
    'age', 'sex', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
    'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
    'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
]

@app.route("/", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        try:
            form_data = request.form.to_dict()
            input_data = {}
            sex_map = {'Male': 1, 'Female': 0}

            for feature in feature_columns:
                if feature in form_data and form_data[feature]:
                    if feature == 'sex':
                        input_data[feature] = sex_map.get(form_data[feature], -1)
                    else:
                        input_data[feature] = float(form_data[feature])
                else:
                    return f"<h3>Error: Missing required field <code>{feature}</code></h3><p><a href='/'>Back</a></p>"

            # Check for invalid sex mapping
            if input_data['sex'] == -1:
                return "<h3>Error: Invalid value for 'sex'. Please select Male or Female.</h3><p><a href='/'>Back</a></p>"

            # === Convert to DataFrame ===
            input_df = pd.DataFrame([input_data])

            # === Drop Correlated Features ===
            input_df.drop(columns=[col for col in to_drop if col in input_df.columns], inplace=True)

            # === Scale Features ===
            input_scaled = scaler.transform(input_df)

            # === Select Features ===
            input_selected = selector.transform(input_scaled)

            # === Predict ===
            prediction = best_model.predict(input_selected)[0]
            probability = best_model.predict_proba(input_selected)[0][1]

            result = "🧠 Parkinson's Disease Detected" if prediction == 1 else "✅ No Parkinson's Disease Detected"
            confidence = f"Confidence: {probability * 100:.2f}%"

            # === Return Result ===
            return f"""
            <h2>Prediction Result</h2>
            <p><b>{result}</b></p>
            <p>{confidence}</p>
            <p><a href="/">Back to Form</a></p>
            """

        except Exception as e:
            return f"<h3>Error: {str(e)}</h3><pre>{traceback.format_exc()}</pre><p><a href='/'>Back to Form</a></p>"

    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)
