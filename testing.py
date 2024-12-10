import pandas as pd
import networkx
import matplotlib.pyplot as plt
    

def run_task():
    data_patients = pd.read_csv("data/patients.csv")
    data_conditions = pd.read_csv("data/conditions.csv")

    unique_conditions = data_conditions['DESCRIPTION'].unique().tolist()
    unique_patients = data_patients['Id'].unique().tolist()

    x_nodes = unique_conditions + unique_patients

    mygraph = networkx.Graph()
    for node_val in x_nodes:
        mygraph.add_node(node_val)

    x_edges = []
    for row_num, condition_row in data_conditions.iterrows():
        condition_patient_id = condition_row['PATIENT']
        condition_desc = condition_row['DESCRIPTION']

        patient_index = x_nodes.index(condition_patient_id)
        condition_index = x_nodes.index(condition_desc)

        row_edge = [patient_index, condition_index]
        x_edges.append(row_edge)

        mygraph.add_edge(condition_patient_id, condition_desc)

    networkx.draw(mygraph, with_labels=True)

if __name__ == '__main__':
    my_dataset = run_task()

