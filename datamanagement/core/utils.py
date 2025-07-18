import json
import logging
from typing import List, Set
from huggingface_hub import login
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

def load_json_links(file_paths: List[str]) -> List[str]:
    """Load and deduplicate URLs from multiple JSON files."""
    seen: Set[str] = set()
    combined: List[str] = []
    for file in file_paths:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                urls = json.load(f)
                for url in urls:
                    if url not in seen:
                        seen.add(url)
                        combined.append(url)
        except Exception as e:
            logger.error(f"Failed to load {file}: {e}")
    return combined

def save_links_to_json(links: List[str], output_file: str):
    """Save list of links to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(links, f, indent=2)
        logger.info(f"Saved {len(links)} links to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save to {output_file}: {e}")

def huggingface_login():
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token)
