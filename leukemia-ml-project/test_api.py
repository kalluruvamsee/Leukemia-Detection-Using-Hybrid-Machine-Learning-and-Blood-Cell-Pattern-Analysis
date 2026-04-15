import requests
import json

url = 'http://localhost:5000/predict'
files = {'file': open(r'C:\Users\kallu\.gemini\antigravity\brain\2baecd62-0c31-4563-a33e-f2585debfebd\sample_leukemia_positive.jpg', 'rb')}


try:
    response = requests.post(url, files=files)
    data = response.json()
    if 'trace' in data:
        with open('error_log.txt', 'w') as f:
            f.write(data['trace'])
        print("Trace written to error_log.txt")
    else:
        print("====== SUCCESS ======")
        print(json.dumps(data, indent=2))
except Exception as e:
    print("FAILED TO CONNECT:", e)
