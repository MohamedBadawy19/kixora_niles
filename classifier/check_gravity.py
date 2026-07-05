import pandas as pd
import numpy as np

print('GRAVITY AXIS COMPARISON')
print('='*50)

# Training data
for name in ['Ap Kanan.xlsx', 'Diam.csv']:
    if name.endswith('.xlsx'):
        df = pd.read_excel('d:/Downloads/Dataset/raw_data/' + name, sheet_name=0)
    else:
        df = pd.read_csv('d:/Downloads/Dataset/raw_data/' + name)
    df.columns = ['X','Y','Z']
    rest = df.head(50)
    print('Training ' + name)
    print('  X mean: %.2f' % rest.X.mean())
    print('  Y mean: %.2f' % rest.Y.mean())
    print('  Z mean: %.2f' % rest.Z.mean())
    print()

# Phone data
cols = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']
for folder, label in [
    ('frontkick(around 12-13 not perfect kicks)', 'Front Kick'),
    ('roundhouse(around 10 or 11 not perfect kicks)', 'Roundhouse'),
    ('standing_and_walking', 'Standing'),
]:
    df = pd.read_csv('d:/Downloads/Dataset/my_records/' + folder + '/Raw Data.csv')
    rest = df.head(250)
    print('Phone ' + label)
    print('  X mean: %.2f' % rest[cols[0]].mean())
    print('  Y mean: %.2f' % rest[cols[1]].mean())
    print('  Z mean: %.2f' % rest[cols[2]].mean())
    print()
