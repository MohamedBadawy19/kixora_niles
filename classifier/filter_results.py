import json

with open(r'd:\Downloads\Dataset\classifier\output\phone_validation_v3.json') as f:
    results = json.load(f)

print('FILTERED RESULTS - only bursts > 0.5 seconds (actual kicks)')
print('='*60)

recordings = {
    'Front Kick (Ap Chagi)': 'Ap Chagi',
    'Roundhouse (Dolyo Chagi)': 'Dolyo Chagi',
    'Standing/Walking (Idle)': 'Idle',
}

total_c, total_n = 0, 0
for rec, expected in recordings.items():
    segs = results[rec]
    if expected != 'Idle':
        real = [s for s in segs if s['duration'] >= 0.5]
    else:
        real = segs

    preds = [s['prediction'] for s in real]
    correct = sum(1 for p in preds if p == expected)
    total_c += correct
    total_n += len(preds)

    print(f'\n{rec} (expected: {expected}):')
    print(f'  Total bursts: {len(segs)} -> After filter: {len(real)}')
    for s in real:
        marker = 'OK' if s['prediction'] == expected else 'WRONG'
        t = s["time"]
        d = s["duration"]
        p = s["prediction"]
        c = s["confidence"]
        print(f'    {t}s ({d}s) -> {p} ({c:.0%}) [{marker}]')
    if preds:
        print(f'  Accuracy: {correct}/{len(preds)} ({correct/len(preds)*100:.0f}%)')

print(f'\n{"="*60}')
if total_n:
    print(f'FILTERED OVERALL: {total_c}/{total_n} ({total_c/total_n*100:.1f}%)')
