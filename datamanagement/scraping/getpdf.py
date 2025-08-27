import os
import requests
class CollectPDFs():
    
    def scrape(self, urls, downloads_folder: str):
        os.makedirs(downloads_folder, exist_ok=True)
        for url in urls:
            pdf_url = url.rstrip().replace(".html", ".pdf")
            filename = os.path.basename(pdf_url)

            try:
                print(f"Downloading: {pdf_url}")
                r = requests.get(pdf_url, stream=True, timeout=60)
                r.raise_for_status()

                if "application/pdf" in r.headers.get("Content-Type", ""):
                    filepath = os.path.join(downloads_folder, filename)
                    with open(filepath, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Saved: {filepath}")
                else:
                    print(f"Skipped (Not PDF): {pdf_url}")

            except Exception as e:
                print(f"Failed: {pdf_url} ({e})")
