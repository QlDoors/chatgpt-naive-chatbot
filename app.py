# -*- coding: utf-8 -*-

__author__ = 'Jude'

import os

import gradio as gr
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, \
    ServiceContext
from langchain.chat_models import ChatOpenAI


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 400
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    index.save_to_disk('index.json')
    return index


def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response


if __name__ == '__main__':
    with open("api-key.txt") as f:
        api_key = f.readline()
        os.environ["OPENAI_API_KEY"] = api_key
        g_index = construct_index("docs")
        iface = gr.Interface(fn=chatbot,
                             inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                             outputs="text",
                             title="Chatgpt Naive Chatbot")
        iface.launch(share=True)
