from flask import Flask, render_template, request, jsonify
import requests
import uuid
from datetime import datetime
import json

app = Flask(__name__)

API_BASE_URL = "http://localhost:8000"  # Your FastAPI service URL

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit_job', methods=['POST'])
def submit_job():
    try:
        # Get user input from the form
        input_text = request.form['input_text']
        
        # Generate a unique identifier for the job (max 25 characters)
        identifier = f"web_user_{uuid.uuid4().hex[:16]}"  # Shortened UUID to fit within 25 characters

        # Prepare the payload for the start_job API
        payload = json.dumps({
            "identifier_from_purchaser": identifier,
            "input_data": {
                "text": input_text
            }
        })

        # Set headers for the API request
        headers = {
            'Content-Type': 'application/json',
            'accept': 'application/json',
            'token': 'iofsnaiojdoiewqajdriknjonasfoinasd'
        }
        url = f"{API_BASE_URL}/start_job"
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an error for bad responses (4xx/5xx)

        # Parse the response from start_job
        start_job_response = response.json()
        job_id = start_job_response["job_id"]
        blockchain_identifier = start_job_response["blockchainIdentifier"]

        # Print the full response on the screen
        print(f"Start Job Response: {start_job_response}")

        # Prepare the payload for the payment API
        payment_payload = {
            "identifierFromPurchaser": identifier,
            "blockchainIdentifier": blockchain_identifier,
            "network": "Preprod",
            "sellerVkey": start_job_response["sellerVkey"],
            "paymentType": "Web3CardanoV1",
            "submitResultTime": start_job_response["submitResultTime"],
            "unlockTime": start_job_response["unlockTime"],
            "externalDisputeUnlockTime": start_job_response["externalDisputeUnlockTime"],
            "agentIdentifier": start_job_response["agentIdentifier"],
            "inputHash": start_job_response["input_hash"],
        }

        # Make the second API call to the payment endpoint
        payment_url = "https://payment.masumi.network/api/v1/purchase/"
        payment_response = requests.post(payment_url, json=payment_payload, headers=headers)
        payment_response.raise_for_status()  # Raise an error for bad responses (4xx/5xx)

        # Return the combined response to the frontend
        return jsonify({
            "status": "success",
            "start_job_response": start_job_response,
            "payment_response": payment_response.json()
        })

    except requests.exceptions.RequestException as e:
        # Handle errors and return a meaningful message
        error_message = str(e.response.json() if e.response else e)
        return jsonify({
            "status": "error",
            "message": error_message
        }), 400 if e.response else 500

@app.route('/check_status/<job_id>')
def check_status(job_id):
    try:
        response = requests.get(f"{API_BASE_URL}/status?job_id={job_id}")
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)