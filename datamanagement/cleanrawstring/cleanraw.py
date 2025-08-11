import re

def clean_for_web_agent(raw: str):
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', raw)
    cleaned = re.sub(r'https?://[^\s)]+', '', cleaned)

    cleaned = re.sub(r'\[.*?\]\((javascript:void\(0\)|/t5/community-help-knowledge-base/community-help/ta-p/4662356|/html/assets/.*?\.pdf)\)', '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'[#]{2,}', '', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    cleaned = cleaned.strip()

    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    cleaned = re.sub(r'USER_AGENT environment variable not set,.*\n', '', cleaned)

    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned)
    cleaned = re.sub(r'\[[A-Za-z \d]+\]\(.*?avatar.*?\)', '', cleaned)
    cleaned = re.sub(r'\[Level \d+\]', '', cleaned)
    cleaned = re.sub(r'(Level \d+).*', '', cleaned)
    cleaned = '\n'.join(line.strip() for line in cleaned.splitlines() if line.strip())
    cleaned = re.sub(r'Discover and save your favorite ideas[\s\S]*$', '', cleaned, flags=re.IGNORECASE)

    return cleaned
