import json
with open('street_types.txt') as fp:
    text = list(fp.read().splitlines(keepends=True))

street_types = {}

for line in text:
    full, abbrv = line.split(' - ')
    street_types.update({full.lower(): abbrv.lower().strip()})

print(street_types)

with open('street_types.json', 'w') as fp:
    json.dump(street_types, fp)
