import sys
import os
from dotenv import load_dotenv

# Set sys.path to project root: /CAI_Webex
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from db.mongo import MongoDBConn
from config.settings import COMMUNITY_FILES, WEBEX_FILE

load_dotenv(dotenv_path=r'datamanagement\core\.env')
MONGO_URI = os.getenv("MONGO_URI")

def load_urls(files):
    seen = set()
    all_urls = []
    for file in files:
        with open(file) as f:
            for url in json.load(f):
                if url not in seen:
                    seen.add(url)
                    all_urls.append(url)
    return all_urls

if __name__ == "__main__":
    final_urls = load_urls(COMMUNITY_FILES)

    with open(WEBEX_FILE) as f:
        webex_urls = json.load(f)

    print("Saving community threads...")
    community = MongoDBConn(
        loginurl=MONGO_URI,
        source="community",
        weburl=final_urls,
        database="cisco_docs",
        collection="dataset"
    )
    community.save_data_to_mongo()

    print("Saving webex articles...")
    webex = MongoDBConn(
        loginurl=MONGO_URI,
        source="webex",
        weburl=webex_urls,
        database="cisco_docs",
        collection="dataset"
    )
    webex.save_data_to_mongo()
