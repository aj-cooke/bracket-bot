import requests
from bs4 import BeautifulSoup
import re
import json

def remove_after_char(s: str, char: str) -> str:
    pattern = re.escape(char) + r".*$"
    return re.sub(pattern, "", s)

url = "https://www.sports-reference.com/cbb/seasons/men/2025-school-stats.html"  # Replace with target URL
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract all links
links = [link.get('href') for link in soup.find_all('a', href=True) if link.get('href').startswith('/cbb/schools/')]
links = list(set([remove_after_char(x[13:], "/").strip() for x in links]))
links = [x for x in links if len(x) > 0]
for link in links:
    print(link)

data = {'schools': links}
with open("Documents/bracket-bot/data/schools.json", "w") as file:
    json.dump(data, file, indent=4)

