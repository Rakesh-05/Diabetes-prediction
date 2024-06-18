import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np

# Step 1: Load and prepare the dataset
data = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\ML CBP\\archive (1)\\diabetes.csv")
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test_scaled)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Step 6: Save the model and scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Step 7: Create the Flask web application
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    
    # Scale the features
    scaled_features = scaler.transform(final_features)
    
    # Predict the outcome
    prediction = model.predict(scaled_features)
    
    output = 'Diabetic' if prediction[0] == 1 else 'Non-diabetic'
    
    return render_template('index.html', prediction_text=f'The person is {output}')

if __name__ == "__main__":
    app.run(debug=True)
