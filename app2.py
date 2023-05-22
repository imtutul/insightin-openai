
from flask import Flask, jsonify, make_response, request
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = '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'

app = Flask(__name__)

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

index = construct_index("docs")

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return make_response(
            jsonify(
                {"success": False,
                 "error": "Unexpected error 2, request is not in JSON format"}),
            400)
    
    try:
        data = request.json
        query = data["data"]
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
        response = index.query(query[0], response_mode="compact")
        return jsonify({"success": True, "data": response.response})
    except:
        return make_response(
            jsonify(
                {"success": False, 
                 "error": "Unexpected error 3: failed to send the message"}),
            400)

if __name__ == '__main__':
   app.run()