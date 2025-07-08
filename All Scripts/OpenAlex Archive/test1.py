import json
import pandas as pd
from striprtf.striprtf import rtf_to_text

# Read and convert the RTF file to plain text
with open('alltopics.rtf', 'r', encoding='utf-8') as f:
    rtf_content = f.read()
plain_text = rtf_to_text(rtf_content)

# Parse the JSON content
data = json.loads(plain_text)
results = data.get('results', [])

# Flatten each item; for nested lists (e.g. keywords, siblings) join the values into strings.
rows = []
for item in results:
    rows.append({
        'id': item.get('id'),
        'display_name': item.get('display_name'),
        'description': item.get('description'),
        'keywords': ', '.join(item.get('keywords', [])),
        'ids_openalex': item.get('ids', {}).get('openalex'),
        'ids_wikipedia': item.get('ids', {}).get('wikipedia'),
        'subfield': item.get('subfield', {}).get('display_name'),
        'field': item.get('field', {}).get('display_name'),
        'domain': item.get('domain', {}).get('display_name'),
        'siblings': ', '.join(s.get('display_name', '') for s in item.get('siblings', [])),
        'works_count': item.get('works_count'),
        'cited_by_count': item.get('cited_by_count'),
        'updated_date': item.get('updated_date'),
        'created_date': item.get('created_date')
    })

# Create a DataFrame and write it to an Excel file
df = pd.DataFrame(rows)
df.to_excel('output.xlsx', index=False)
