import argparse

import deepeval.api
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from uptrain import Settings
from DeepEvalLocalModel import DeepEvalLocalModel
from llm import get_llm

load_tokenizer = []
llm_args = {"context_window": 4096, "max_new_tokens": 256,
                             "generate_kwargs": {"temperature": 0.7, "top_k": 50, "top_p": 0.95}}
def qwen_completion_to_prompt(completion):
    tokenizer = load_tokenizer[0]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": completion}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
class EvalModelAgent():
    def __init__(self, args):
        parser = argparse.ArgumentParser(description='RAG-benchmark-evaluate')
        parser.add_argument('--llamaIndexEvaluateModel', type=str, default="Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", help='llamaIndex local model')
        parser.add_argument('--deepEvalEvaluateModel', type=str, default="Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", help='DeepEval local model')
        parser.add_argument('--upTrainEvaluateModel', type=str, default="qwen:7b-chat-v1.5-q8_0", help='Uptrain local model')
        parser.add_argument('--evaluateApiName', type=str, default="", help='api name')
        parser.add_argument('--evaluateApiKey', type=str, default="", help='api key')

        self.args = args
        llamaIndex_LocalmodelName = self.args.llamaIndexEvaluateModel
        deepEval_LocalModelName = self.args.deepEvalEvaluateModel
        uptrain_LocalModelName = self.args.upTrainEvaluateModel
        api_name = self.args.evaluateApiName
        api_key = self.args.evaluateApiKey
        if api_name == "":
            self._llama_model = AutoModelForCausalLM.from_pretrained(llamaIndex_LocalmodelName,
                                                     torch_dtype=torch.float16,
                                                     device_map="auto").eval()
            self._llama_tokenizer = AutoTokenizer.from_pretrained(llamaIndex_LocalmodelName)
            load_tokenizer.append(self._llama_tokenizer)
            self.llamaModel = HuggingFaceLLM(context_window=llm_args["context_window"],
                              max_new_tokens=llm_args["max_new_tokens"],
                              completion_to_prompt=qwen_completion_to_prompt,
                              generate_kwargs=llm_args["generate_kwargs"],
                              model=self._llama_model,
                              tokenizer=self._llama_tokenizer,
                              device_map="cuda:0",)
        else:
            self.llamaModel = OpenAI(api_key=api_key, api_base="https://uiuiapi.com/v1",
                      model=api_name)
        if api_name == "":
            if deepEval_LocalModelName == llamaIndex_LocalmodelName:
                self._deepEval_model = self._llama_model
                self._deepEval_tokenizer = self._llama_tokenizer
            else:
                self._deepEval_model = AutoModelForCausalLM.from_pretrained(deepEval_LocalModelName,
                                                     torch_dtype=torch.float16,
                                                     device_map="auto").eval()
                self._deepEval_tokenizer = AutoTokenizer.from_pretrained(deepEval_LocalModelName)
            self.deepEvalModel = DeepEvalLocalModel(model=self._deepEval_model,
                                                    tokenizer=self._deepEval_tokenizer)
        else:
            deepeval.api.API_BASE_URL = 'https://uiuiapi.com/v1'
            self.deepEvalModel = api_name
        if api_name == "":
            self.uptrainSetting = Settings(model="ollama/"+uptrain_LocalModelName)
        else:
            self.uptrainSetting = Settings(
                    model=api_name,
                    openai_api_key=api_key,
                    base_url="https://uiuiapi.com/v1",
                )