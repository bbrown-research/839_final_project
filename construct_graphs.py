import argparse
import copy

import networkx
import numpy as np
import pandas as pd
from openai import OpenAI

from build_graph_from_csv import make_graph, Data
from eval import edit_dist, spectral_dist
from gpt_prompting import prompt_gpt_model


def copy_over_some_nodes(add_to_graph, source_graph, copy_percent=0.1):
    target_nodes = set(list(add_to_graph.nodes))
    usable_source_nodes = set(list(source_graph.nodes))

    nodes_in_src_not_in_tgt = usable_source_nodes.difference(target_nodes)
    only_patients_not_in = [x for x in nodes_in_src_not_in_tgt if '-' in x or ' ' in x]

    num_to_sample = round(len(only_patients_not_in) * copy_percent)
    sampled_patients = np.random.choice(only_patients_not_in, size=num_to_sample, replace=False)

    new_target_graph = copy.deepcopy(add_to_graph)
    for p in sampled_patients:
        new_target_graph.add_node(p)

    return new_target_graph, sampled_patients


def copy_over_some_edges(add_to_graph, source_graph, new_nodes, copy_percent):
    all_added_edges = []
    all_missing_edges = []
    for node in new_nodes:
        source_edges = list(source_graph.edges(node))
        added_edges = np.array(source_edges)[np.random.choice(list(range(len(source_edges))), size=round(copy_percent*len(source_edges)), replace=False)].tolist()
        added_edges = set([tuple(x) for x in added_edges])
        missing_edges = set(source_edges).difference(added_edges)

        all_added_edges += list(added_edges)
        all_missing_edges += list(missing_edges)

        for edge in added_edges:
            add_to_graph.add_edge(edge[0], edge[1])
    return add_to_graph, all_added_edges, all_missing_edges

if __name__ == '__main__':
    np.random.seed(12345)

    train_graph, train_data, train_id_map = make_graph(get_filename=Data.get_train, hash_type='np_nt', replace_id_w_name=True)
    test_graph, test_data, test_id_map = make_graph(get_filename=Data.get_test, hash_type='np_nt', replace_id_w_name=True)
    combined_graph, _, _ = make_graph(get_filename=Data.get_all, hash_type='np_nt', replace_id_w_name=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_key", help='Your OpenAI api key')
    args = parser.parse_args()

    ############################################# Experiment 1 #############################################
    #Add some nodes from the test set
    exp1_graph, copied_nodes = copy_over_some_nodes(add_to_graph=train_graph, source_graph=test_graph, copy_percent=1.0)
    # Add back the some of the edges for some of the nodes from the test set

    mutated_nodes_percent = 0.1
    nodes_to_add_some_edges = list(np.random.choice(copied_nodes, size=round(mutated_nodes_percent*len(copied_nodes)), replace=False))
    nodes_to_add_all_edges = set(copied_nodes).difference(set(nodes_to_add_some_edges))
    exp1_graph_partial_nodes_and_edges, added_edges1, _ = copy_over_some_edges(add_to_graph=copy.deepcopy(exp1_graph),
                                                                                             source_graph=test_graph,
                                                                                             new_nodes=nodes_to_add_all_edges,
                                                                                             copy_percent=1.0)
    exp1_graph_with_added_nodes_and_edges, added_edges2, missing_edges = copy_over_some_edges(add_to_graph=copy.deepcopy(exp1_graph_partial_nodes_and_edges),
                                                                                             source_graph=test_graph,
                                                                                             new_nodes=nodes_to_add_some_edges,
                                                                                             copy_percent=0.1)

    # add missing nodes to the exp1_graph_with_added_nodes_and_edges for distance calculation. These nodes are in the test data, but are on
    # excluded/removed edges, so the model won't be able to predict them (they are not in training set)
    nodes_for_which_no_training_data_exists = combined_graph.nodes - exp1_graph_with_added_nodes_and_edges.nodes
    for node in nodes_for_which_no_training_data_exists:
        exp1_graph_with_added_nodes_and_edges.add_node(node)

    #add missing nodes to the exp1_graph for distance calculation
    base_graph_missing = (combined_graph.nodes - exp1_graph_partial_nodes_and_edges.nodes)
    for missing_node in base_graph_missing:
        exp1_graph_partial_nodes_and_edges.add_node(missing_node)

    desired_fields = ['Id', 'STATE', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME', 'GENDER', 'RACE', 'ETHNICITY', 'BIRTHDATE']
    train_id_map.update(test_id_map)

    train_patients_dat = pd.read_csv('data/train_patients.csv')[desired_fields]
    test_patients_dat = pd.read_csv('data/test_patients.csv')[desired_fields]
    all_dat = pd.concat([train_patients_dat, test_patients_dat], axis=0)
    all_dat['NAME'] = all_dat['Id'].apply(lambda x: train_id_map[x])
    all_dat = all_dat.drop('Id', axis=1)
    all_dat['AGE'] = all_dat['BIRTHDATE'].apply(lambda x: 2024-int(x.split("-")[0]))
    all_dat = all_dat.drop('BIRTHDATE', axis=1)
    patient_info = "\n".join([f"{', '.join([f'{all_dat.columns[j]}: {str(z)}' for j,z in enumerate(x)])}" for _,x in all_dat.iterrows()])

    nodes_text = ", ".join(nodes_to_add_some_edges)
    graph_text = ["(" + ", ".join(x) + ")" for x in list(exp1_graph_with_added_nodes_and_edges.edges)]
    graph_text = "\n".join(graph_text)

    add_max_n_edges = False
    max_n_edges_text = ""
    if add_max_n_edges:
        max_n_edges_text = f"At most, only {len(missing_edges)} edges are missing from the graph. Make sure your output is less that {len(missing_edges)} edges, " \
                           f"although the actual number of missing edges may be closer to {len(missing_edges)//2} or even {len(missing_edges)//10}."

    gpt_query = f"I will give you a partially completed knowledge graph that I want you to complete. The knowledge graph connects information about patients " \
                f"with allergies, conditions, medications, and health observations that are described by SNOMED-CT codes. Patients are the nodes with names like " \
                f"Bob123 Fletcher456 or Alice473 Sanchez789. The graph I will give " \
                f"you is formatted as a list of edges, where each edge is described as a tuple of start and end nodes, for example (start_node, end_node). " \
                f"I will also give you information about each patient that you can use. Specifically, I want you to figure out for the list of patients " \
                f"with missing edges that I give you which set of patients in the graph are most similar. You can use those similar patients to predict the missing " \
                f"edges for the patients with missing edges. Additionally, for non-patient nodes, use the node name, which is a number like 856987, as a SNOMED-CT " \
                f"code. SNOMED-CT codes give information about that node that you can use to reason about which patients might be connected to which nodes. " \
                f"Given the graph and a list of patients who have missing edges, I want you to predict which edges are missing for the list of patients " \
                f"with missing edges you are given. Only output a list of tuples that describe the edges that are missing and do not add any new nodes to the graph, " \
                f" for example: (A, B) where A and B are both nodes that already exist in the graph.\n" \
                f"\n" \
                f"The information about the patients: {patient_info}\n" \
                f"\n" \
                f"The nodes that have edges missing are: {nodes_text}\n" \
                f"\n" \
                f"The graph, specified by tuples of edges in (Node A, Node B) format is: {graph_text}.\n" \
                f"\n" \
                f"For nodes that have missing edges that I provided to you, please predict which edges are missing. Give your output as a list of " \
                f"edges only, and do not add any additional text. Your output should look like: (Node A, Node B)\n (Node A, Node C)\n (Node D, Node E), " \
                f"etc...\n " \
                f"{max_n_edges_text}"


    open_ai_client = OpenAI(api_key=args.openai_key)
    prompt_result = prompt_gpt_model(open_ai_client,
                                     gpt_query,
                                     model_type="gpt-4o-mini",
                                     attachments=None,
                                     max_output_tokens=2000,
                                     max_timeout_ms=1000)

    #eval
    gpt_added_edges_txt = prompt_result.replace("```plaintext", "").strip().split("\n")
    gpt_added_edges_txt = [x for x in gpt_added_edges_txt if '(' in x and ')' in x]
    gpt_added_edges = [x.replace("(","").replace(")","").split(", ") for x in gpt_added_edges_txt]

    for edge in gpt_added_edges:
        exp1_graph_with_added_nodes_and_edges.add_edge(edge[0], edge[1])

    hallucinated_nodes = exp1_graph_with_added_nodes_and_edges.nodes() - combined_graph.nodes()
    for node in hallucinated_nodes:
        exp1_graph_with_added_nodes_and_edges.remove_node(node)
    print(f"Number of GPT hallucinated nodes: {len(hallucinated_nodes)}")

    #Replacing graphs for metrics calc
    ordered_combo = networkx.Graph()
    ordered_combo.add_nodes_from(sorted(combined_graph.nodes(data=True)))
    ordered_combo.add_edges_from(combined_graph.edges(data=True))

    ordered_partial = networkx.Graph()
    ordered_partial.add_nodes_from(sorted(exp1_graph_partial_nodes_and_edges.nodes(data=True)))
    ordered_partial.add_edges_from(exp1_graph_partial_nodes_and_edges.edges(data=True))

    exp1_with_addons = networkx.Graph()
    exp1_with_addons.add_nodes_from(sorted(exp1_graph_with_added_nodes_and_edges.nodes(data=True)))
    exp1_with_addons.add_edges_from(exp1_graph_with_added_nodes_and_edges.edges(data=True))

    #Getting distances
    dist_edit_combo_to_gpt = edit_dist(G1=ordered_combo, G2=exp1_with_addons)
    dist_s_combo_to_gpt = spectral_dist(G1=ordered_combo, G2=exp1_with_addons, p=2, kind="laplacian")

    dist_edit_combo_to_noedge = edit_dist(G1=ordered_combo, G2=ordered_partial)
    dist_s_combo_to_noedge = spectral_dist(G1=ordered_combo, G2=ordered_partial, p=2, kind="laplacian")

    print(f"Experiment 01 Edit Distance: {dist_edit_combo_to_gpt}")
    print(f"Experiment 01 Spectral Distance: {dist_s_combo_to_gpt}")

    print(f"Experiment 01 Edit Distance from Graph with no edges added: {dist_edit_combo_to_noedge}")
    print(f"Experiment 01 Spectral Distance from Graph with no edges added: {dist_s_combo_to_noedge}")

    print(f"Num Nodes with some edges missing: {len(nodes_to_add_some_edges)} and total number of edges missing: {len(missing_edges)}")

    false_positive = 0
    true_positive = 0
    not_recovered = 0
    for x in gpt_added_edges:
        if x in missing_edges:
            true_positive += 1
        else:
            false_positive += 1
    for x in missing_edges:
        if x not in gpt_added_edges:
            not_recovered += 1

    TPR = true_positive / len(gpt_added_edges)
    FPR = false_positive / len(gpt_added_edges)
    NRR = not_recovered / len(missing_edges)

    print(f"Edges in GPT Correct Rate: {TPR} from true_positive / len(gpt_added_edges) = {true_positive}/{len(gpt_added_edges)}")
    print(f"Edges Hallucinated by GPT Rate: {FPR} from false_positive / len(gpt_added_edges) = {false_positive}/{len(gpt_added_edges)}")
    print(f"Edges Not Recovered by GPT Rate: {NRR} from not_recovered / len(missing_edges) = {not_recovered}/{len(missing_edges)}")