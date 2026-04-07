import requests
import xml.etree.ElementTree as ET

HEADERS = {
    'User-Agent': 'Justin Sanford justin.sanford@dlhcorp.com',
    'Accept-Encoding': 'gzip, deflate',
}

cik = '1061768'
accession = '000156761924000363'
url = (f'https://www.sec.gov/Archives/edgar/data/'
       f'{cik}/{accession}/form13fInfoTable.xml')

r = requests.get(url, headers=HEADERS)
print(f"XML status: {r.status_code}")
print(f"Content length: {len(r.content)}")
print(f"First 500 chars:")
print(r.content[:500])