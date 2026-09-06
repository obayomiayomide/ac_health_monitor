from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import joblib
import os
import json
from datetime import datetime
from collections import deque
import onnxruntime as ort
import csv
# import ai_edge_litert.interpreter as litert
# import tensorflow as tf

app = Flask(__name__)

# ============================================================
# LOAD MODELS
# ============================================================
last_esp32_seen = None

print("Loading models...")

try:
    rf_model = joblib.load('rf_model.joblib')
    scaler   = joblib.load('scaler.joblib')
    le       = joblib.load('label_encoder.joblib')
    RF_READY = True
    print("✓ Random Forest loaded")
except Exception as e:
    RF_READY = False
    print(f"⚠ RF not loaded: {e}")

try:
    # lstm_model = tf.keras.models.load_model('lstm_model.h5')
    # LSTM_READY = True
    # print("✓ LSTM loaded")
    
    # interpreter = litert.Interpreter(model_path='lstm_model.tflite')
    # interpreter.allocate_tensors()
    # print("✓ LSTM TFLite loaded (Standard fallback)")

    # Load the ONNX model session
    ort_session = ort.InferenceSession('lstm_model.onnx')
    
    # Get input and output names required by ONNX to route data
    lstm_input_name = ort_session.get_inputs()[0].name
    lstm_output_name = ort_session.get_outputs()[0].name
        
    LSTM_READY = True
    print("✓ LSTM ONNX loaded")
except Exception as e:
    LSTM_READY = False
    print(f"⚠ LSTM not loaded: {e}")

# ============================================================
# IN-MEMORY BUFFER
# Stores last 60 readings for LSTM sequence
# and last 100 readings for dashboard chart history
# ============================================================
sensor_buffer   = deque(maxlen=60)   # LSTM needs 60 timesteps
history_buffer  = deque(maxlen=100)  # dashboard chart history

# def predict_lstm_tflite(features_seq):
#     input_details  = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     input_data = np.array(features_seq, dtype=np.float32).reshape(1, 60, 10)
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()
#     proba = interpreter.get_tensor(output_details[0]['index'])[0]
#     pred  = le.inverse_transform([np.argmax(proba)])[0]
#     conf  = round(float(max(proba)) * 100, 1)
#     return pred, conf

def predict_lstm_onnx(features_seq):
    # Prepare your input data exactly as you did before (shape: 1, 60, 10)
    input_data = np.array(features_seq, dtype=np.float32).reshape(1, 60, 10)
    
    # Run inference via ONNX Runtime
    outputs = ort_session.run([lstm_output_name], {lstm_input_name: input_data})
    
    # Extract probabilities (ONNX returns a list of outputs, grab the first one)
    proba = outputs[0][0] 
    
    # Keep the rest of your original post-processing logic exactly the same
    pred = le.inverse_transform([np.argmax(proba)])[0]
    conf = round(float(max(proba)) * 100, 1)
    
    return pred, conf

FEATURE_COLS = [
    'supply_temp_C', 'room_temp_C', 'temp_differential_C',
    'vibration_magnitude', 'vibration_std', 'gyroscope',
    'current_A', 'low_pressure_PSI', 'high_pressure_PSI',
    'compressor_on'
]

# ============================================================
# ALERT LOGIC
# ============================================================
ALERT_RULES = {
    'HEALTHY': {
        'level': 'green',
        'message': 'System operating normally.',
        'recommendation': 'No action required.'
    },
    'LOW_REFRIGERANT': {
        'level': 'yellow',
        'message': 'Low refrigerant charge detected.',
        'recommendation': 'Schedule refrigerant top-up with a certified technician. Monitor cooling performance.'
    },
    'BLOCKED_CONDENSER': {
        'level': 'orange',
        'message': 'Condenser blockage detected.',
        'recommendation': 'Clean the outdoor unit coils. Remove debris from around the unit. Check airflow.'
    },
    'BEARING_WEAR': {
        'level': 'red',
        'message': 'Compressor bearing wear detected.',
        'recommendation': 'Stop AC use immediately. Contact a technician urgently to inspect the compressor.'
    },
    'INTERMITTENT_COMPRESSOR': {
        'level': 'orange',
        'message': 'Intermittent compressor cycling detected.',
        'recommendation': 'Check capacitor and contactor. Inspect electrical connections. Schedule urgent service.'
    },
}

# ============================================================
# CONFIRMATION FILTER
# Require 3 consecutive predictions before alerting
# Prevents false positives from single noisy readings
# ============================================================
prediction_history = deque(maxlen=3)

def confirmed_prediction(new_pred):
    prediction_history.append(new_pred)
    if len(prediction_history) == 3 and len(set(prediction_history)) == 1:
        return new_pred
    return list(prediction_history)[-1]

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/manifest.json')
def manifest():
    return jsonify({
        "name": "AC Health Monitor",
        "short_name": "AC Monitor",
        "description": "AI-based predictive maintenance for AC/HVAC systems",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#1D9E75",
        "orientation": "portrait-primary",
        "icons": [
            {"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png"}
        ]
    })

# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# static_js_dir = os.path.join(BASE_DIR, 'static', 'js')

@app.route('/sw.js')
def service_worker():
    return send_from_directory(
        'static', 
        'js/sw.js', 
        mimetype='application/javascript'
    )

@app.route('/predict', methods=['POST'])
def predict():
    global last_esp32_seen
    
    # Check if request comes from ESP32 (no source=demo query param)
    is_demo = request.args.get('is_demo') == 'true'
    if not is_demo:
        last_esp32_seen = datetime.now()

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        # Extract and compute features
        supply_temp  = float(data.get('supply_temp', 0))
        room_temp    = float(data.get('room_temp', 0))
        temp_diff    = round(room_temp - supply_temp, 3)
        vib_mag      = float(data.get('vibration_mag', 0))
        vib_std      = float(data.get('vibration_std', 0))
        gyro         = float(data.get('gyroscope', 0))
        current      = float(data.get('current', 0))
        low_psi      = float(data.get('low_pressure', 0))
        high_psi     = float(data.get('high_pressure', 0))
        comp_on      = int(data.get('compressor_on', 1))

        # ✅ ADD LOGGING BLOCK HERE — right after feature extraction
        # if not is_demo:
        #     with open('real_healthy_data.csv', 'a', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow([
        #             datetime.now().strftime('%H:%M:%S'),
        #             supply_temp, room_temp, temp_diff,
        #             vib_mag, vib_std, gyro,
        #             current, low_psi, high_psi,
        #             comp_on, 'HEALTHY'
        #         ])

        features = [
            supply_temp, room_temp, temp_diff,
            vib_mag, vib_std, gyro,
            current, low_psi, high_psi, comp_on
        ]

        # Add to buffers
        sensor_buffer.append(features)
        if not is_demo: 
            history_buffer.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'supply_temp': supply_temp,
                'room_temp': room_temp,
                'temp_diff': temp_diff,
                'vib_mag': vib_mag,
                'current': current,
                'low_psi': low_psi,
                'high_psi': high_psi,
            })

        # --- Random Forest Prediction ---
        rf_pred = 'UNAVAILABLE'
        rf_conf = 0.0
        if RF_READY:
            x = np.array(features).reshape(1, -1)
            x_scaled = scaler.transform(x)
            rf_pred = le.inverse_transform(rf_model.predict(x_scaled))[0]
            rf_conf = round(float(max(rf_model.predict_proba(x_scaled)[0])) * 100, 1)

        # --- LSTM Prediction ---
        lstm_pred = 'UNAVAILABLE'
        lstm_conf = 0.0
        if LSTM_READY and len(sensor_buffer) == 60:
            x_seq = np.array(list(sensor_buffer))
            x_seq_scaled = scaler.transform(x_seq)
            x_seq_scaled = x_seq_scaled.reshape(1, 60, len(FEATURE_COLS))

            # proba = lstm_model.predict(x_seq_scaled, verbose=0)[0]
            # lstm_pred = le.inverse_transform([np.argmax(proba)])[0]
            # lstm_conf = round(float(max(proba)) * 100, 1)

            # lstm_pred, lstm_conf = predict_lstm_tflite(x_seq_scaled)
            # Call your newly optimized ONNX prediction function
            lstm_pred, lstm_conf = predict_lstm_onnx(x_seq_scaled)

        # Use RF if LSTM buffer not filled yet
        primary_pred = lstm_pred if LSTM_READY and len(sensor_buffer) == 60 else rf_pred
        final_pred   = confirmed_prediction(primary_pred)

        # Get alert info
        alert = ALERT_RULES.get(final_pred, ALERT_RULES['HEALTHY'])

        return jsonify({
            'rf_prediction':    rf_pred,
            'rf_confidence':    rf_conf,
            'lstm_prediction':  lstm_pred,
            'lstm_confidence':  lstm_conf,
            'final_prediction': final_pred,
            'alert_level':      alert['level'],
            'alert_message':    alert['message'],
            'recommendation':   alert['recommendation'],
            'buffer_size':      len(sensor_buffer),
            'timestamp':        datetime.now().strftime('%H:%M:%S'),
            'features': {
                'supply_temp': supply_temp,
                'room_temp': room_temp,
                'temp_diff': temp_diff,
                'vib_mag': vib_mag,
                'current': current,
                'low_psi': low_psi,
                'high_psi': high_psi,
                'compressor_on': comp_on
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    return jsonify(list(history_buffer))


@app.route('/status', methods=['GET'])
def status():
    # Check if an ESP32 posted in the last 10 seconds
    has_esp32 = False
    if last_esp32_seen:
        has_esp32 = (datetime.now() - last_esp32_seen).total_seconds() < 10
    
    return jsonify({
        'rf_ready':   RF_READY,
        'lstm_ready': LSTM_READY,
        'esp32_connected': has_esp32, # <--- Return actual hardware state
        'buffer':     len(sensor_buffer),
        'history':    len(history_buffer),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
