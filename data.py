import urllib.request
import ssl
import gzip, json

# Download Electronics 5-core reviews
url = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
with opener.open(url) as resp, open("Electronics_5.json.gz", "wb") as f:
    f.write(resp.read())

# Read and preview
with gzip.open("Electronics_5.json.gz") as f:
    for i, line in enumerate(f):
        review = json.loads(line)
        print(review)
        if i == 4: break  # preview first 5 rows


