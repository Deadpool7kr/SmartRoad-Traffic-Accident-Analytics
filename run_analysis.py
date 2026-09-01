from pathlib import Path
import argparse, sys
sys.path.insert(0, str(Path(__file__).parent/'src'))
from analytics import run_pipeline

p=argparse.ArgumentParser()
p.add_argument('--data',default='data/road_collisions_2025.csv')
p.add_argument('--output',default='outputs')
p.add_argument('--model',default='models/smartroad_model.joblib')
a=p.parse_args()
if not Path(a.data).exists():
    demo='data/demo_accidents.csv'
    print(f'Input {a.data} not found; using DEMO dataset {demo} for validation.')
    a.data=demo
summary,_,imp,_=run_pipeline(a.data,a.output,a.model)
print('Validation complete.')
print(summary['metrics'])
print('Top features:')
print(imp.head(10).to_string(index=False))
