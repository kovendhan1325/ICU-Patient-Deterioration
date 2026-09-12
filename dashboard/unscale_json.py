import json
import os

# Define realistic clinical means and standard deviations for inverse scaling
# Formula: Unscaled = (Scaled * Std) + Mean
CLINICAL_STATS = {
    'heartrate':        {'mean': 80,    'std': 15},
    'systemicsystolic': {'mean': 115,   'std': 20},
    'systemicdiastolic':{'mean': 70,    'std': 12},
    'systemicmean':     {'mean': 85,    'std': 15},
    'respiration':      {'mean': 18,    'std': 4},
    'sao2':             {'mean': 97,    'std': 2},
    'temperature':      {'mean': 37.0,  'std': 0.7},
    'Creatinine':       {'mean': 1.0,   'std': 0.5},
    'Glucose':          {'mean': 110,   'std': 30},
    'Lactate':          {'mean': 1.5,   'std': 1.0},
    'Sodium':           {'mean': 140,   'std': 4},
    'Potassium':        {'mean': 4.0,   'std': 0.5},
    'WBC':              {'mean': 8.0,   'std': 3.0},
    'Hemoglobin':       {'mean': 13.0,  'std': 2.0},
    'Platelets':        {'mean': 250,   'std': 60},
    'BUN':              {'mean': 15,    'std': 7}
}

file_path = 'patients_data.json'

print(f"Loading {file_path}...")
with open(file_path, 'r') as f:
    data = json.load(f)

patients = data.get('patients', [])
print(f"Found {len(patients)} patients to process.")

def unscale(val, key):
    if val is None:
        return None
    stats = CLINICAL_STATS.get(key)
    if not stats:
        return val
    
    # Inverse transform
    real_val = (val * stats['std']) + stats['mean']
    
    # Ensure no physically impossible values due to extreme outliers
    if key == 'sao2':
        real_val = min(100.0, max(50.0, real_val))
    elif key in ['heartrate', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'respiration']:
        real_val = max(1.0, real_val)
        
    return round(real_val, 2)

for p in patients:
    # Unscale latest vitals
    vitals = p.get('vitals', {})
    for k, v in vitals.items():
        vitals[k] = unscale(v, k)
    
    # Unscale timeline
    timeline = p.get('timeline', [])
    for obs in timeline:
        for k in list(obs.keys()):
            if k == 'hour':
                continue
            obs[k] = unscale(obs[k], k)

print("Saving unscaled data back...")
with open(file_path, 'w') as f:
    json.dump(data, f)
    
print("Done! Data has been successfully unscaled to real physiological ranges.")
