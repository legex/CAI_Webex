import json
from typing import List, Set
from huggingface_hub import login
import os
from dotenv import load_dotenv
from core.logger import setup_logger

# Setup logger for this module
logger = setup_logger("huggingface_utils", 'log/huggingface_utils.log')
load_dotenv()

def load_json_links(file_paths: List[str]) -> List[str]:
    """
    Load and deduplicate URLs from multiple JSON files.

    Args:
        file_paths (List[str]): List of JSON file paths each containing a list of URLs.

    Returns:
        List[str]: Combined list of unique URLs loaded from all files.
    """
    seen: Set[str] = set()
    combined: List[str] = []
    for file in file_paths:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                urls = json.load(f)
                count_before = len(combined)
                for url in urls:
                    if url not in seen:
                        seen.add(url)
                        combined.append(url)
                logger.info(f"Loaded {len(urls)} URLs from {file}, {len(combined) - count_before} new unique URLs added.")
        except Exception as e:
            logger.error(f"Failed to load {file}: {e}")
    logger.info(f"Total unique URLs combined: {len(combined)}")
    return combined

def save_links_to_json(links: List[str], output_file: str):
    """
    Save a list of links to a JSON file.

    Args:
        links (List[str]): List of URLs to save.
        output_file (str): Path of the output JSON file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(links, f, indent=2)
        logger.info(f"Saved {len(links)} links to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save to {output_file}: {e}")

def huggingface_login():
    """
    Log in to Hugging Face Hub using token from environment variable 'HUGGINGFACE_TOKEN'.

    Loads environment variables, retrieves the token, and logs into huggingface_hub.
    """
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        try:
            login(token)
            logger.info("Successfully logged in to Hugging Face Hub.")
        except Exception as e:
            logger.error(f"Failed to login to Hugging Face: {e}")
    else:
        logger.warning("HUGGINGFACE_TOKEN not found in environment variables; skipping login.")
