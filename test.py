import requests
import json

url = "http://localhost:8000/start_job"

payload = json.dumps({
  "identifier_from_purchaser": "example_purchaser_123",
  "input_data": {
    "text": "I am Minion and I am looking for a house. I want to buy/build a house where the focus is on wood work. I have the total budget of 100K and want the house in the next two years. I am open to +10% increase in my budget if my preferences are fulfilled, but if not then try to keep the budget as minimum as possible."
  }
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
