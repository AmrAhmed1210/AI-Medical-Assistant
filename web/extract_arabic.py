import os
import re
import json

directory = r"C:\Users\user\AI-Medical-Assistant\web\src"
arabic_pattern = re.compile(r'[\u0600-\u06FF\s]+')
# Actually we want strings that contain any arabic character, maybe mixed with punctuation
# Match any string literal that contains arabic characters
# e.g. 'مرحبا', "مرحبا بك", `مرحبا`
regex = re.compile(r'([\'"\`])(.*?(?:[\u0600-\u06FF]).*?)\1')
jsx_text_regex = re.compile(r'>\s*([^<]*?[\u0600-\u06FF][^<]*?)\s*<')

results = set()

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith(('.tsx', '.ts')):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Find in quotes
                for match in regex.findall(content):
                    results.add(match[1].strip())
                # Find in JSX text
                for match in jsx_text_regex.findall(content):
                    results.add(match.strip())

with open("arabic_strings.json", "w", encoding="utf-8") as f:
    json.dump(list(results), f, ensure_ascii=False, indent=2)

print(f"Extracted {len(results)} unique strings.")
