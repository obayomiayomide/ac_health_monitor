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

devices = {}

# ============================================================
# LOAD MODELS
# ============================================================

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

def get_or_create_device(device_id):
    """Initializes isolated memory buffers if a new ESP32 connects."""
    if device_id not in devices:
        devices[device_id] = {
            'last_seen': None,
            'latest_prediction': None,
            'sensor_buffer': deque(maxlen=60),
            'history_buffer': deque(maxlen=100) # Adjust history capacity as needed
        }
    return devices[device_id]

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
    input_data = np.array(features_seq, dtype=np.float32).reshape(1, 60, 9)
    
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
    'current_A', 'low_pressure_PSI',
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
    return render_template('landing.html')

@app.route('/simulator')
def simulator():
    return render_template('simulator.html')

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
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        # Extract the Unique Device ID sent from the ESP32
        # Default to a generic ID if testing from a demo/web environment
        device_id = data.get('device_id', 'demo_device')
        dev = get_or_create_device(device_id)

        is_demo = request.args.get('is_demo') == 'true' or device_id == 'demo_device'
        if not is_demo:
            dev['last_seen'] = datetime.now()

        # Extract and compute features
        supply_temp  = float(data.get('supply_temp', 0))
        room_temp    = float(data.get('room_temp', 0))
        temp_diff    = round(room_temp - supply_temp, 3)
        vib_mag      = float(data.get('vibration_mag', 0))
        vib_std      = float(data.get('vibration_std', 0))
        gyro         = float(data.get('gyroscope', 0))
        current      = float(data.get('current', 0))
        low_psi      = float(data.get('low_pressure', 0))
        comp_on      = int(data.get('compressor_on', 1))

        # ✅ Isolated data logging (Appends unique Device ID inside CSV file)
        # if not is_demo:
        #     with open('real_healthy_data_v2_2.csv', 'a', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow([
        #             datetime.now().strftime('%H:%M:%S'),
        #             device_id, # Added to know which machine logged it
        #             supply_temp, room_temp, temp_diff,
        #             vib_mag, vib_std, gyro,
        #             current, low_psi,
        #             comp_on, 'HEALTHY'
        #         ])

        features = [
            supply_temp, room_temp, temp_diff,
            vib_mag, vib_std, gyro,
            current, low_psi, comp_on
        ]

        # Add to this specific device's buffer space
        dev['sensor_buffer'].append(features)
        if not is_demo: 
            dev['history_buffer'].append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'supply_temp': supply_temp,
                'room_temp': room_temp,
                'temp_diff': temp_diff,
                'vib_mag': vib_mag,
                'current': current,
                'low_psi': low_psi,
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
        if LSTM_READY and len(dev['sensor_buffer']) == 60:
            x_seq = np.array(list(dev['sensor_buffer']))
            x_seq_scaled = scaler.transform(x_seq)
            x_seq_scaled = x_seq_scaled.reshape(1, 60, len(FEATURE_COLS))
            lstm_pred, lstm_conf = predict_lstm_onnx(x_seq_scaled)

        # Use RF if LSTM buffer not filled yet
        primary_pred = lstm_pred if LSTM_READY and len(dev['sensor_buffer']) == 60 else rf_pred
        final_pred   = confirmed_prediction(primary_pred)

        # Get alert info
        alert = ALERT_RULES.get(final_pred, ALERT_RULES['HEALTHY'])

        # Store isolated calculations back into the device object
        dev['latest_prediction'] = {
            'rf_prediction':    rf_pred,
            'rf_confidence':    rf_conf,
            'lstm_prediction':  lstm_pred,
            'lstm_confidence':  lstm_conf,
            'final_prediction': final_pred,
            'alert_level':      alert['level'],
            'alert_message':    alert['message'],
            'recommendation':   alert['recommendation'],
            'buffer_size':      len(dev['sensor_buffer']),
            'timestamp':        datetime.now().strftime('%H:%M:%S'),
            'features': {
                'supply_temp': supply_temp,
                'room_temp': room_temp,
                'temp_diff': temp_diff,
                'vib_mag': vib_mag,
                'current': current,
                'low_psi': low_psi,
                'compressor_on': comp_on
            }
        }

        return jsonify(dev['latest_prediction'])

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate-device/<device_id>', methods=['GET'])
def validate_device(device_id):
    # Check if the device exists and has actually recorded data
    if device_id in devices and devices[device_id]['latest_prediction'] is not None:
        return jsonify({'exists': True}), 200 # 200 = Success, it exists!
    
    # Change this to 404 Not Found
    return jsonify({'error': 'Device not found or inactive'}), 404 

@app.route('/api/<device_id>/get-latest', methods=['GET'])
def get_latest(device_id):
    if device_id not in devices or not devices[device_id]['latest_prediction']:
        return jsonify({'status': 'no_data'}), 200
    return jsonify(devices[device_id]['latest_prediction'])

@app.route('/api/<device_id>/history', methods=['GET'])
def history(device_id):
    if device_id not in devices:
        return jsonify([])
    return jsonify(list(devices[device_id]['history_buffer']))


@app.route('/api/<device_id>/status', methods=['GET'])
def status(device_id):
    if device_id not in devices:
        return jsonify({'rf_ready':   RF_READY,
                'lstm_ready': LSTM_READY,'esp32_connected': False, 'buffer': 0, 'history':0})
        
    dev = devices[device_id]
    has_esp32 = False
    if dev['last_seen']:
        has_esp32 = (datetime.now() - dev['last_seen']).total_seconds() < 10
    
    return jsonify({
        'rf_ready':   RF_READY,
        'lstm_ready': LSTM_READY,
        'esp32_connected': has_esp32,
        'buffer':     len(dev['sensor_buffer']),
        'history':    len(dev['history_buffer']),
    })

# --- NEW ENTRYWAY DASHBOARD ROUTE ---
@app.route('/dashboard/<device_id>')
def dashboard_view(device_id):
    # This renders your existing dashboard.html but injects the unique ID context
    return render_template('dashboard.html', device_id=device_id)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
