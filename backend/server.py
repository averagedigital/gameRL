from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# File to store the best brain
BRAIN_FILE = 'best_brain.json'

# Initialize brain file if not exists
if not os.path.exists(BRAIN_FILE):
    with open(BRAIN_FILE, 'w') as f:
        # Save a dummy structure or empty
        json.dump({"fitness": 0, "brain": None}, f)

@app.route('/get_brain', methods=['GET'])
def get_brain():
    try:
        with open(BRAIN_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/post_brain', methods=['POST'])
def post_brain():
    try:
        new_data = request.json
        new_fitness = new_data.get('fitness', 0)
        
        # Read current best
        with open(BRAIN_FILE, 'r') as f:
            current_data = json.load(f)
            
        current_fitness = current_data.get('fitness', 0)
        
        # Only update if better (strict improvement)
        if new_fitness > current_fitness:
            print(f"New Record! {new_fitness} > {current_fitness}")
            with open(BRAIN_FILE, 'w') as f:
                json.dump(new_data, f)
            return jsonify({"status": "updated", "new_record": new_fitness})
        else:
            return jsonify({"status": "ignored", "current_record": current_fitness})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "GameRL Backend is Running!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

