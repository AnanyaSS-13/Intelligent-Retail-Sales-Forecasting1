from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load("model(1).pkl")
store_encoder = joblib.load("store_encoder (5).pkl")
item_encoder = joblib.load("item_encoder (4).pkl")

# Create Flask app
app = Flask(__name__)
# Home page
@app.route('/')
def home():
    return render_template('index.html')
# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        store_id = int(request.form['store_id'])
        item_id = int(request.form['item_id'])
        price = float(request.form['price'])
        promo = int(request.form['promo'])
        weekday = int(request.form['weekday'])
        month = int(request.form['month'])
        year = int(request.form['year'])
        day = int(request.form['day'])

        # Prepare input for prediction
        input_data = np.array([[store_id, item_id, price, promo, weekday, month, year, day]])

        # Predict sales
        prediction = model.predict(input_data)

        return render_template(
            'index.html',
            prediction_text=f'Predicted Sales: {prediction[0]:.2f} Units'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )
if __name__ == "__main__":
    app.run(debug=True)