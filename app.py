from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

# Create the FastAPI app
app = FastAPI()

# Generate a synthetic dataset for demonstration purposes
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Define the input data model using Pydantic
class ParkinsonInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float


@app.get("/")
def read_root():
    return {"message": "Welcome to the Parkinson Detection API"}


@app.post("/predict")
def predict(input_data: ParkinsonInput):
    # Prepare the input data for the model
    data = np.array([[input_data.feature1, input_data.feature2, input_data.feature3, input_data.feature4]])

    # Make a prediction
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    # Return the result
    return {
        "prediction": int(prediction[0]),  # 0 or 1
        "probability": probability[0].tolist()  # Probabilities for each class
    }