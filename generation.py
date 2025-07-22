# from apigateway.services.rag_engine import RagEngine

# rg = RagEngine()
# while True:
#     print("\n\n-------------------------------")
#     question = input("Ask your question (q to quit): ")
#     print("\n\n")
#     if question == "q":
#         break

#     response = rg.generate_response(question)
#     print(response)

from datamanagement.scraping.link_collector import LinkCollector
import json

website = "https://community.cisco.com/t5/webex-community/ct-p/webex-user"

lc = LinkCollector("community",website)

listofurls = lc.scrape_website_community(max_pages=450)

with open("link_metadata5.json", 'w') as f:
    json.dump(listofurls, f)


pdfscrapeurls = ["https://www.cisco.com/c/en/us/support/unified-communications/unified-communications-manager-callmanager/products-maintenance-guides-list.html", ]