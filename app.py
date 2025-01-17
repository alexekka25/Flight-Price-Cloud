from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model and preprocessing objects
model = pickle.load(open('flight_price_model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = request.form
    input_data = {
        'airline': label_encoders['airline'].transform([data['airline']])[0],
        'source_city': label_encoders['source_city'].transform([data['source_city'].strip().lower()])[0],
        'departure_time': label_encoders['departure_time'].transform([data['departure_time'].strip().lower()])[0],
        'stops': label_encoders['stops'].transform([data['stops'].strip().lower()])[0],
        'arrival_time': label_encoders['arrival_time'].transform([data['arrival_time'].strip().lower()])[0],
        'destination_city': label_encoders['destination_city'].transform([data['destination_city'].strip().lower()])[0],
        'class': label_encoders['class'].transform([data['class'].strip().lower()])[0],
        'duration': float(data['duration']),
        'days_left': int(data['days_left']),
    }
    
    # Scale numerical features
    scaled_features = scaler.transform([[input_data['duration'], input_data['days_left']]])
    input_data['duration'], input_data['days_left'] = scaled_features[0]

    # Prepare input for prediction
    features = np.array([list(input_data.values())])
    prediction = model.predict(features)

    # Format the prediction
    predicted_price = '{:,.2f}'.format(prediction[0])

    return '''
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f0f0f0;
            }}
            .container {{
                text-align: center;
            }}
            .price {{
                font-size: 48px;
                font-weight: bold;
            }}
            .price .currency, .price .value {{
                color: blue;
            }}
            .plane {{
                width: 50px;
                position: relative;
                margin-bottom: 20px;
                animation: fly 3s infinite linear;
            }}
            @keyframes fly {{
                0% {{ transform: translateX(-100px); }}
                50% {{ transform: translateX(50px); }}
                100% {{ transform: translateX(-100px); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="plane"></div>
            <p class="price"><span class="currency">Rs</span> <span class="value">{}</span></p>
        </div>
    </body>
    </html>
    '''.format(predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
