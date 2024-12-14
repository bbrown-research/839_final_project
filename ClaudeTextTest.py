 %%
# Import necessary libraries
import pandas as pd
import networkx as nx
import json
import random
from typing import List, Dict, Tuple
import itertools
import os

def load_data():
    try:
        # Load all CSV files
        patients_df = pd.read_csv('/content/patients.csv')
        print(f"Loaded patients data with {len(patients_df)} rows")
        conditions_df = pd.read_csv('/content/conditions.csv')
        print(f"Loaded conditions data with {len(conditions_df)} rows")
        medications_df = pd.read_csv('/content/medications.csv')
        print(f"Loaded medications data with {len(medications_df)} rows")
        observations_df = pd.read_csv('/content/observations.csv')
        print(f"Loaded observations data with {len(observations_df)} rows")
        allergies_df = pd.read_csv('/content/allergies.csv')
        print(f"Loaded allergies data with {len(allergies_df)} rows")

        return patients_df, conditions_df, medications_df, observations_df, allergies_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def create_knowledge_graph(patients_df, conditions_df, medications_df, observations_df, allergies_df):
    G = nx.DiGraph()

    # Add patient nodes
    for _, patient in patients_df.iterrows():
        G.add_node(patient['Id'],
                  type='patient',
                  attributes={
                      'gender': patient['GENDER'],
                      'birthdate': patient['BIRTHDATE'],
                      'race': patient['RACE']
                  })

    # Add conditions with better error handling
    for _, condition in conditions_df.iterrows():
        try:
            condition_id = f"condition_{condition['CODE']}"
            condition_desc = str(condition['DESCRIPTION']).lower()
            G.add_node(condition_id,
                      type='condition',
                      attributes={'description': condition_desc})
            G.add_edge(condition['PATIENT'], condition_id,
                      relation_type='has_condition',
                      start_date=condition['START'],
                      stop_date=condition['STOP'])
        except Exception as e:
            print(f"Error adding condition: {str(e)}")

    # Add medications with better error handling
    for _, medication in medications_df.iterrows():
        try:
            med_id = f"medication_{medication['CODE']}"
            med_desc = str(medication['DESCRIPTION']).lower()
            G.add_node(med_id,
                      type='medication',
                      attributes={'description': med_desc})
            G.add_edge(medication['PATIENT'], med_id,
                      relation_type='takes_medication',
                      start_date=medication['START'],
                      stop_date=medication['STOP'])
        except Exception as e:
            print(f"Error adding medication: {str(e)}")

    print(f"Created graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

def find_medical_patterns(G):
    patterns = []

    # Define common medical patterns with more flexible matching
    common_patterns = {
        'diabetes': {
            'conditions': ['diabetes', 'glucose', 'sugar'],
            'medications': ['insulin', 'metformin', 'glipizide'],
            'observations': ['glucose', 'hba1c', 'sugar']
        },
        'hypertension': {
            'conditions': ['hypertension', 'high blood pressure'],
            'medications': ['lisinopril', 'hydrochlorothiazide', 'amlodipine'],
            'observations': ['blood pressure', 'sodium']
        }
    }

    patient_count = 0
    for patient in [n for n, d in G.nodes(data=True) if d['type'] == 'patient']:
        patient_count += 1
        neighbors = list(G.neighbors(patient))

        for pattern_name, pattern in common_patterns.items():
            matches = {
                'conditions': [],
                'medications': [],
                'observations': []
            }

            for neighbor in neighbors:
                node_data = G.nodes[neighbor]
                node_desc = str(node_data['attributes'].get('description', '')).lower()

                if 'condition' in node_data['type']:
                    for keyword in pattern['conditions']:
                        if keyword in node_desc:
                            matches['conditions'].append(node_desc)
                            break

                elif 'medication' in node_data['type']:
                    for keyword in pattern['medications']:
                        if keyword in node_desc:
                            matches['medications'].append(node_desc)
                            break

            if matches['conditions'] or matches['medications']:
                patterns.append({
                    'pattern_name': pattern_name,
                    'patient_id': patient,
                    'matches': matches
                })

    print(f"Processed {patient_count} patients")
    print(f"Found {len(patterns)} pattern matches")
    return patterns

def generate_test_prompts(patterns: List[Dict]):
    prompts = []

    for pattern in patterns:
        if not (pattern['matches']['conditions'] or pattern['matches']['medications']):
            continue

        # Create training example
        training_example = f"""
Example Pattern:
Patient with {pattern['pattern_name']}:
Conditions: {', '.join(pattern['matches']['conditions']) if pattern['matches']['conditions'] else 'None'}
Medications: {', '.join(pattern['matches']['medications']) if pattern['matches']['medications'] else 'None'}
"""

        # Create test case
        if pattern['matches']['medications']:
            test_case = {
                'given': f"""
Patient has the following conditions:
{', '.join(pattern['matches']['conditions']) if pattern['matches']['conditions'] else 'None'}

Question: What medications would you expect to find in their record?
""",
                'hidden': pattern['matches']['medications']
            }

            prompt = f"""
{training_example}

New case:
{test_case['given']}

Please provide:
1. Your inferences about what medications should be in the record
2. Your confidence level (high/medium/low)
3. Your reasoning based on the example pattern and medical knowledge

Ground truth (for evaluation): {', '.join(test_case['hidden'])}
-------------------
"""
            prompts.append(prompt)

    print(f"Generated {len(prompts)} test prompts")
    return prompts

def main():
    print("\nStarting data loading process...")
    dfs = load_data()

    print("\nCreating knowledge graph...")
    G = create_knowledge_graph(*dfs)

    print("\nFinding medical patterns...")
    patterns = find_medical_patterns(G)

    print("\nGenerating test prompts...")
    prompts = generate_test_prompts(patterns)

    print("\n=== GENERATED PROMPTS ===\n")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n=== Prompt {i} ===")
        print(prompt)
        print("\n")

if __name__ == "__main__":
    main()

# %%
def generate_training_contexts(G, sizes=[5, 10, 15]):
    """Generate training contexts of different sizes"""
    training_contexts = {}

    patient_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'patient']

    for size in sizes:
        context = ["=== Medical Knowledge Graph Training Context (Size: {size}) ===\n"]
        selected_patients = patient_nodes[:size]

        for patient in selected_patients:
            patient_info = ["Patient Case:"]
            conditions = []
            medications = []
            observations = []

            # Get all directly connected nodes
            for neighbor in G.neighbors(patient):
                node_data = G.nodes[neighbor]
                edge_data = G.edges[patient, neighbor]

                if node_data.get('type') == 'condition':
                    conditions.append((node_data['attributes']['description'],
                                    edge_data.get('start_date', '')))
                elif node_data.get('type') == 'medication':
                    medications.append((node_data['attributes']['description'],
                                     edge_data.get('start_date', '')))
                elif node_data.get('type') == 'observation':
                    observations.append((node_data['attributes']['description'],
                                      edge_data.get('start_date', '')))

            if conditions:
                patient_info.append("Conditions:")
                for cond, date in conditions:
                    patient_info.append(f"- {cond} (onset: {date})")

            if medications:
                patient_info.append("Medications:")
                for med, date in medications:
                    patient_info.append(f"- {med} (started: {date})")

            if observations:
                patient_info.append("Observations:")
                for obs, date in observations:
                    patient_info.append(f"- {obs} (recorded: {date})")

            context.append("\n".join(patient_info) + "\n")

        training_contexts[size] = "\n".join(context)

    return training_contexts

def generate_test_cases(G, training_patients):
    """Generate test cases from non-training patients"""
    test_cases = []

    # Get patients not in training set
    test_patients = [n for n, d in G.nodes(data=True)
                    if d.get('type') == 'patient' and n not in training_patients]

    for patient in test_patients:
        # Get patient's full context
        conditions = []
        medications = []
        observations = []

        for neighbor in G.neighbors(patient):
            node_data = G.nodes[neighbor]
            edge_data = G.edges[patient, neighbor]

            if node_data.get('type') == 'condition':
                conditions.append(node_data['attributes']['description'])
            elif node_data.get('type') == 'medication':
                medications.append(node_data['attributes']['description'])
            elif node_data.get('type') == 'observation':
                observations.append(node_data['attributes']['description'])

        # Create test cases by hiding different pieces of information
        if conditions and medications:
            # Hide a condition
            for condition in conditions:
                remaining_conditions = [c for c in conditions if c != condition]
                test_case = {
                    'type': 'missing_condition',
                    'context': {
                        'conditions': remaining_conditions,
                        'medications': medications,
                        'observations': observations
                    },
                    'answer': condition
                }
                test_cases.append(test_case)

            # Hide a medication
            for medication in medications:
                remaining_medications = [m for m in medications if m != medication]
                test_case = {
                    'type': 'missing_medication',
                    'context': {
                        'conditions': conditions,
                        'medications': remaining_medications,
                        'observations': observations
                    },
                    'answer': medication
                }
                test_cases.append(test_case)

    return test_cases

def format_test_case(test_case):
    """Format a single test case into a prompt"""
    context = test_case['context']

    prompt = "Test Case:\n"
    if context['conditions']:
        prompt += f"Conditions: {'; '.join(context['conditions'])}\n"
    if context['medications']:
        prompt += f"Medications: {'; '.join(context['medications'])}\n"
    if context['observations']:
        prompt += f"Observations: {'; '.join(context['observations'])}\n"

    prompt += f"\nQuestion: The patient is missing a medication or condition. Based on the medical context above, what missing {test_case['type'].replace('missing_', '')} would you expect to find in this patient's record?\n"
    prompt += """
Please provide:
1. Your prediction
2. Confidence level (high/medium/low)
3. Reasoning based on similar patterns in the training cases
4. Specific examples from the training cases that support your prediction
"""

    return prompt

def save_training_contexts(training_contexts, output_dir='training_contexts'):
   """Save each training context to a separate file"""
   import os

   # Create directory if it doesn't exist
   if not os.path.exists(output_dir):
       os.makedirs(output_dir)

   # Save each context to a separate file
   for size, context in training_contexts.items():
       filename = os.path.join(output_dir, f'training_context_size_{size}.txt')
       with open(filename, 'w', encoding='utf-8') as f:
           f.write(context)
       print(f"Saved training context size {size} to {filename}")

def save_test_cases(test_cases, output_dir='test_cases', cases_per_file=234):
    """Save test cases into multiple files based on type, complexity, and size constraints"""
    import os
    import math

    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First categorize by type and complexity
    categories = {
        'condition_to_medication_simple': [],
        'condition_to_medication_complex': [],
        'medication_to_condition_simple': [],
        'medication_to_condition_complex': []
    }

    # Categorize test cases
    for test_case in test_cases:
        context = test_case['context']
        total_elements = len(context['conditions']) + len(context['medications'])
        is_complex = total_elements > 10

        if test_case['type'] == 'missing_medication':
            if is_complex:
                categories['condition_to_medication_complex'].append(test_case)
            else:
                categories['condition_to_medication_simple'].append(test_case)
        elif test_case['type'] == 'missing_condition':
            if is_complex:
                categories['medication_to_condition_complex'].append(test_case)
            else:
                categories['medication_to_condition_simple'].append(test_case)

    # Save files and collect ground truths
    all_ground_truths = []

    for category, cases in categories.items():
        if not cases:
            continue

        # Split category into multiple files if needed
        num_files = math.ceil(len(cases) / cases_per_file)

        for file_num in range(num_files):
            start_idx = file_num * cases_per_file
            end_idx = min((file_num + 1) * cases_per_file, len(cases))
            current_cases = cases[start_idx:end_idx]

            # Create unique file identifier
            file_id = f"{category}_part_{file_num + 1}"
            test_file = os.path.join(output_dir, f'test_cases_{file_id}.txt')

            # Save test cases
            with open(test_file, 'w', encoding='utf-8') as f:
                for i, test_case in enumerate(current_cases, 1):
                    case_id = f"{file_id}_{i}"  # Unique identifier for each case
                    f.write(f"\n=== Test Case {case_id} ===\n")
                    f.write(format_test_case(test_case))
                    f.write("\n" + "="*50 + "\n")

            print(f"Saved {len(current_cases)} test cases to {test_file}")

            # Collect ground truths with identifiers
            for i, test_case in enumerate(current_cases, 1):
                case_id = f"{file_id}_{i}"
                all_ground_truths.append({
                    'case_id': case_id,
                    'file': f'test_cases_{file_id}.txt',
                    'category': category,
                    'case_number': i,
                    'answer': test_case['answer']
                })

    # Save ground truths with clear mapping
    ground_truth_file = os.path.join(output_dir, 'ground_truths.txt')
    with open(ground_truth_file, 'w', encoding='utf-8') as f:
        f.write("# Ground Truth Reference File\n")
        f.write("# Format: case_id | file | category | case_number | answer\n")
        f.write("#" + "="*80 + "\n\n")

        for truth in all_ground_truths:
            f.write(f"Case ID: {truth['case_id']}\n")
            f.write(f"File: {truth['file']}\n")
            f.write(f"Category: {truth['category']}\n")
            f.write(f"Case Number: {truth['case_number']}\n")
            f.write(f"Answer: {truth['answer']}\n")
            f.write("-"*40 + "\n")

    print(f"Saved ground truths with mappings to {ground_truth_file}")

def main():
    print("\nStarting data loading process...")
    dfs = load_data()

    print("\nCreating knowledge graph...")
    G = create_knowledge_graph(*dfs)

    print("\nGenerating training contexts of different sizes...")
    training_contexts = generate_training_contexts(G, sizes=[5, 10, 15])

    print("\nGenerating test cases...")
    all_patient_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'patient']
    training_patients = set(all_patient_nodes[:15])
    test_cases = generate_test_cases(G, training_patients)

    # Create directories
    if not os.path.exists('training_contexts'):
        os.makedirs('training_contexts')
    if not os.path.exists('test_cases'):
        os.makedirs('test_cases')

    # Save to files
    print("\nSaving training contexts...")
    save_training_contexts(training_contexts)

    print("\nSaving test cases and ground truths...")
    save_test_cases(test_cases)

if __name__ == "__main__":
    main()
