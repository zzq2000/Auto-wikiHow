# for arg configuration
import argparse

API_KEY = 'sk-NX84YsnF5W8JxX18D62wyyRgKMmAkikWgg5FsETkRvLMQ74W'
AUTH_TOKEN = "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"
API_BASE = 'https://api.chatanywhere.tech/v1'

class cfg():
    def __init__(self):
        self.llm = "chatgpt-3.5"
        #self.llm = "qwen7_int8"
        # self.api_key = 'sk-NX84YsnF5W8JxX18D62wyyRgKMmAkikWgg5FsETkRvLMQ74W'
        self.auth_token = "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"
        # self.api_base = 'https://api.chatanywhere.tech/v1'
        self.embeddings = 'BAAI/bge-large-en-v1.5'
        self.split_type = 'sentence'
        self.chunk_size = 1024
        #self.chunk_size = 128
        self.dataset = 'hotpot_qa'
        self.source_dir = "../wiki"
        self.persist_dir = "../storage"
        #evaluate
        self.llamaIndexEvaluateModel = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8"
        self.deepEvalEvaluateModel = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8"
        self.upTrainEvaluateModel = "qwen:7b-chat-v1.5-q8_0"
        self.evaluateApiName = ""
        self.evaluateApiKey = ""

    def get_args(self):
        parser = argparse.ArgumentParser(description='RAG-benchmark')
        parser.add_argument('--llm', type=str, default=self.llm, help='llm model')
        # parser.add_argument('--api_key', type=str, default=self.api_key, help='api key')
        # parser.add_argument('--auth_token', type=str, default=self.api_key, help='auth token')
        # parser.add_argument('--api_base', type=str, default=self.api_base, help='api base')
        parser.add_argument('--embeddings', type=str, default=self.embeddings, help='embeddings model')
        parser.add_argument('--split_type', type=str, default=self.split_type, help='split type')
        parser.add_argument('--chunk_size', type=int, default=self.chunk_size, help='chunk size')
        parser.add_argument('--dataset', type=str, default=self.dataset, help='dataset')
        parser.add_argument('--source_dir', type=str, default=self.source_dir, help='source directory')
        parser.add_argument('--persist_dir', type=str, default=self.persist_dir, help='persist directory')
        parser.add_argument('--llamaIndexEvaluateModel', type=str, default="Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", help='llamaIndex local model')
        parser.add_argument('--deepEvalEvaluateModel', type=str, default="Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", help='DeepEval local model')
        parser.add_argument('--upTrainEvaluateModel', type=str, default="qwen:7b-chat-v1.5-q8_0", help='Uptrain local model')
        parser.add_argument('--evaluateApiName', type=str, default="", help='api name')
        parser.add_argument('--evaluateApiKey', type=str, default="", help='api key')
        self.args = parser.parse_args()
        self.update()

    def update(self):
        self.llm = self.args.llm
        # self.api_key = self.args.api_key
        # self.api_base = self.args.api_base
        # self.auth_token = self.args.auth_token
        self.embeddings = self.args.embeddings
        self.split_type = self.args.split_type
        self.chunk_size = self.args.chunk_size
        self.dataset = self.args.dataset
        self.source_dir = self.args.source_dir
        self.persist_dir = self.args.persist_dir
        self.llamaIndex_LocalmodelName = self.args.llamaIndexEvaluateModel
        self.deepEval_LocalModelName = self.args.deepEvalEvaluateModel
        self.uptrain_LocalModelName = self.args.upTrainEvaluateModel
        self.api_name = self.args.evaluateApiName
        self.api_key = self.args.evaluateApiKey
