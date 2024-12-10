import json

import networkx
import numpy
from openai import OpenAI
import argparse

from openai.types.beta.threads.message import Attachment
from openai.types.beta.threads.message_create_params import AttachmentToolFileSearch

#PARAMS FOR GPT CONTROL
MAX_OUTPUT_TOKENS = 200
MAX_TIMEOUT_MS = 1000

def prompt_gpt(file_mechanism):
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_key", help='Your OpenAI api key')
    parser.add_argument("--json_prompt_file", help='A JSON file of prompts and filepaths')
    args = parser.parse_args()

    openai_key = args.openai_key
    open_ai_client = OpenAI(api_key=openai_key)

    with open(args.json_prompt_file, 'r') as f:
        json_prompts_and_files = json.load(f)

    prompt_result_information= {}
    for prompt_id, prompt_file_tuple in json_prompts_and_files.items():
        prompt_text = prompt_file_tuple['content']

        file_attachments = None
        if 'file_attachment' in prompt_file_tuple.keys() and prompt_file_tuple['file_attachment'] is not None:
            if file_mechanism == 'attach_file_to_prompt':
                prompt_file_attachment_path = prompt_file_tuple['file_attachment']
                file_attachments = create_gpt_attachment_file(open_ai_client, prompt_file_attachment_path, read_type='rb')
            elif file_mechanism == 'append_to_prompt':
                prompt_file_attachment_path = prompt_file_tuple['file_attachment']
                file_text = open(prompt_file_attachment_path, "r").read().replace(",", ", ")
                prompt_text = prompt_text + f"\n The content is: \n {file_text}"
            else:
                raise NotImplementedError("File processing mechanism '{}' is not implemented.")


        prompt_result = prompt_gpt_model(open_ai_client, prompt_text, model_type="gpt-4o-mini", attachments=file_attachments)

        prompt_result_dict = {
            'input_prompt': prompt_text,
            'prompt_file_attachment_path': prompt_file_attachment_path,
            'result': prompt_result
        }

        prompt_result_information[prompt_id] = prompt_result_dict

    return prompt_result_information


def prompt_gpt_model(open_ai_client, prompt_text, model_type="gpt-4o-mini", attachments=None):
    gpt_assistant_model = open_ai_client.beta.assistants.create(
        model=model_type,
        description="A model that can handle PDF files.",
        tools=[{"type": "file_search"}],
        name="main_model",
    )

    thread = open_ai_client.beta.threads.create()

    if attachments is not None:
        open_ai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            attachments=[attachments],
            content=prompt_text,
        )
    else:
        open_ai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt_text,
        )

    query = open_ai_client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=gpt_assistant_model.id,
        timeout=MAX_TIMEOUT_MS,
        max_completion_tokens=MAX_OUTPUT_TOKENS
    )

    messages_cursor = open_ai_client.beta.threads.messages.list(thread_id=thread.id)
    messages = [message for message in messages_cursor]

    res_txt = messages[0].content[0].text.value

    return res_txt


def create_gpt_attachment_file(open_ai_client, file_path, read_type='rb'):
    file = open_ai_client.files.create(file=open(file_path, read_type), purpose="assistants")
    gpt_attachment =  Attachment(
            file_id=file.id, tools=[AttachmentToolFileSearch(type="file_search")]
        )
    return gpt_attachment


def convert_prompt_results_to_graph(prompt_results):
    for prompt_id, prompt_info in prompt_results.items():
        prompt_result_text = prompt_info['result']
        raw_res = prompt_result_text.split("\n")
        raw_res = [x.strip().replace("(","").replace(")","").split(", ") for x in raw_res]
        edge_matrix = numpy.array(raw_res)

        unique_ids = set(numpy.unique(edge_matrix[:, 0]).tolist() + edge_matrix[:, 1].unique().tolist())

        mygraph = networkx.Graph()
        for node_val in unique_ids:
            mygraph.add_node(node_val)
        for edge in edge_matrix:
            mygraph.add_edge(edge[0], edge[1])

        networkx.draw(mygraph, with_labels=True)


if __name__ == '__main__':
    allowed_file_mechanisms = ['append_to_prompt', 'attach_file_to_prompt']

    #Note: attach_file_to_prompt does not work for CSVs
    prompt_results = prompt_gpt(file_mechanism=allowed_file_mechanisms[0])

    convert_prompt_results_to_graph(prompt_results)
