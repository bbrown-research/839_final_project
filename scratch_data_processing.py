import numpy
import pandas

data = pandas.read_csv("data/conditions.csv")

used_cols = ['START', 'STOP', 'PATIENT', 'DESCRIPTION']
save_data = []

unique_patient_ids = data['PATIENT'].unique()
patient_id_map = {id:patient_num for patient_num, id in enumerate(unique_patient_ids)}

for i, row in data.iterrows():
    row = row[used_cols]
    if '(disorder)' in row['DESCRIPTION']:
        row['PATIENT'] = patient_id_map[row['PATIENT']]
        row['DESCRIPTION'] = row['DESCRIPTION'].replace(" (disorder)", "")

        if row['STOP'] is numpy.nan:
            row['STOP'] = 'No Stop Date'
        save_data.append(row)

final_pd = pandas.DataFrame(save_data, columns=used_cols)
final_pd.to_csv('testing_conditions.txt', header=None, index=None, sep=',')