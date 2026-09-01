import argparse
from pathlib import Path
import requests

URL = 'https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-{year}.csv'

p=argparse.ArgumentParser(description='Download official DfT road collision data')
p.add_argument('--year',type=int,default=2025)
p.add_argument('--output',default=None)
a=p.parse_args()
out=Path(a.output or f'data/road_collisions_{a.year}.csv')
out.parent.mkdir(parents=True,exist_ok=True)
url=URL.format(year=a.year)
print(f'Downloading {url}')
with requests.get(url,stream=True,timeout=120) as r:
    r.raise_for_status()
    with out.open('wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk: f.write(chunk)
print(f'Saved to {out} ({out.stat().st_size/1024/1024:.1f} MB)')
