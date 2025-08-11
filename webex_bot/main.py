from webexteamssdk import WebexTeamsAPI, Webhook
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r'datamanagement\core\.env')
API_TOKEN = os.getenv("WEBEX_TOKEN")

webexapi = WebexTeamsAPI(access_token=API_TOKEN)
import requests
import json

url = "https://webexapis.com/v1/webhooks"
ngrok_url = "https://a88678128cd0.ngrok-free.app"

payload = json.dumps({
  "resource": "messages",
  "event": "all",
  "targetUrl": f"{ngrok_url}/webexhook",
  "name": "MytestwebHook"
})
headers = {
  'Authorization': f'Bearer {API_TOKEN}',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
