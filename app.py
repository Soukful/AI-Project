from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model and preprocessors
model = joblib.load('final_random_forest_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
categorical_cols = joblib.load('categorical_cols.pkl')

# All model input columns (original + encoded)
full_feature_list = joblib.load('original_feature_names.pkl')

# Define default values for remaining features
default_values = {
    'capital-gains': 0,
    'capital-loss': 0,
    'auto_make': 'Subaru',
    'auto_model': 'Legacy',
    'insured_education_level': 'JD',
    'insured_occupation': 'doctor',
    'insured_relationship': 'husband',
    'insured_sex': 'MALE',
    'insured_education_level': 'JD',
    'incident_state': 'SC',
    'incident_city': 'Columbus',
    'incident_type': 'Single Vehicle Collision',
    'incident_severity': 'Minor Damage',
    'collision_type': 'Rear Collision',
    'authorities_contacted': 'Police',
    'policy_state': 'OH',
    'policy_csl': '250/500',
    'insured_hobbies': 'chess',
    'number_of_vehicles_involved': 1,
    'witnesses': 0,
    'umbrella_limit': 0,
    'injury_claim': 0,
    'property_claim': 0,
    'vehicle_claim': 0,
    'incident_hour_of_the_day': 12,
    'age': 45,
    # Add more default values if needed
}

@app.route('/', methods=['GET'])
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')

    # Collect form input
    form_input = request.form.to_dict()

    # Combine form input with defaults
    input_data = {**default_values, **form_input}

    # Create DataFrame
    df_input = pd.DataFrame([input_data])

    # Ensure all required categorical columns exist
    for col in categorical_cols:
        if col not in df_input.columns:
            df_input[col] = default_values.get(col, 'unknown')

    # Split and transform
    X_cat = encoder.transform(df_input[categorical_cols])
    X_num = df_input.drop(columns=categorical_cols).astype(float).values
    X_combined = np.hstack((X_num, X_cat.toarray()))

    # Ensure the feature count matches what StandardScaler expects
    expected_features = scaler.mean_.shape[0]
    current_features = X_combined.shape[1]

    if current_features < expected_features:
        missing_cols = expected_features - current_features
        X_combined = np.hstack([X_combined, np.zeros((1, missing_cols))])


    # Scale and reduce
    X_scaled = scaler.transform(X_combined)
    X_pca = pca.transform(X_scaled)

    # Predict
    prediction = model.predict(X_pca)[0]
    result = "Fraudulent" if prediction == 1 else "Not Fraudulent"

    return render_template('form.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
