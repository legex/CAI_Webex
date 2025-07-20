from webexteamssdk import WebexTeamsAPI, Webhook
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r'datamanagement\core\.env')
API_TOKEN = os.getenv("WEBEX_TOKEN")

webexapi = WebexTeamsAPI(access_token=API_TOKEN)
import requests
import json

url = "https://webexapis.com/v1/webhooks"

payload = json.dumps({
  "resource": "messages",
  "event": "all",
  "targetUrl": "https://e1bdc28f9606.ngrok-free.app/webexhook",
  "name": "MytestwebHook"
})
headers = {
  'Authorization': f'Bearer {API_TOKEN}',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
