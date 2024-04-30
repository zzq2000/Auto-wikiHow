import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import openai
from llama_index_client import SentenceSplitter

from llama_index.core import Settings, PromptTemplate, SimpleDirectoryReader, VectorStoreIndex
from llm import get_llm
from index import get_index
from evaluate import evaluating
from embedding import get_embedding
from qa_loader import get_qa_dataset
from config import cfg
from retriever import *
from evaluate import EvaluationResult
from EvalModelAgent import EvalModelAgent
import random
import numpy as np
import torch
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)
name = "https://api.chatanywhere.com.cn/v1"
auth_token = "sk-NX84YsnF5W8JxX18D62wyyRgKMmAkikWgg5FsETkRvLMQ74W"

cfg = cfg()
cfg.get_args()
qa_dataset = get_qa_dataset(cfg.dataset)
llm = get_llm(cfg.llm)

# Create and dl embeddings instance
embeddings = get_embedding(cfg.embeddings)

Settings.chunk_size = cfg.chunk_size
Settings.llm = llm
Settings.embed_model = embeddings
# pip install llama-index-embeddings-langchain

cfg.persist_dir = cfg.persist_dir + '-' + cfg.dataset + '-' + cfg.embeddings + '-' + cfg.split_type + '-' + str(cfg.chunk_size)

index = get_index(qa_dataset, cfg.persist_dir, split_type=cfg.split_type, chunk_size=cfg.chunk_size)

query_engine = RetrieverQueryEngine(
    retriever=vector_retriever(index),
    response_synthesizer=response_synthesizer(0),
    node_postprocessors=[LongContextReorder()]
)

text_qa_template_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)
text_qa_template = PromptTemplate(text_qa_template_str)

refine_template_str = (
    "We have the opportunity to refine the original answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question: {query_str}. "
    "If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)
refine_template = PromptTemplate(refine_template_str)

# Setup index query engine using LLM
# query_engine = index.as_query_engine(response_mode="compact")

query_engine.update_prompts({"response_synthesizer:text_qa_template": text_qa_template,
                             "response_synthesizer:refine_template": refine_template})
query_engine = query_expansion([query_engine], query_number=4, similarity_top_k=10)
query_engine = RetrieverQueryEngine.from_args(query_engine)


# question = "Which team does the player named 2015 Diamond Head Classic’s MVP play for?"
# answer = "Sacramento Kings"
# golden_source = "The 2015 Diamond Head Classic was a college:    basketball tournament ... Buddy Hield was named the tournament’s MVP. Chavano Rainier ”Buddy” Hield is a Bahamian professional basketball player for the Sacramento Kings of the NBA..."
true_num = 0
all_num = 0
evaluateResults = EvaluationResult(metrics=["Llama_retrieval_Faithfulness", "Llama_retrieval_Relevancy", "Llama_response_correctness",
                              "Llama_response_semanticSimilarity", "Llama_response_answerRelevancy","Llama_retrieval_RelevancyG",
                              "Llama_retrieval_FaithfulnessG",
                              "DeepEval_retrieval_contextualPrecision","DeepEval_retrieval_contextualRecall",
                              "DeepEval_retrieval_contextualRelevancy","DeepEval_retrieval_faithfulness",
                              "DeepEval_response_answerRelevancy","DeepEval_response_hallucination",
                              "DeepEval_response_bias","DeepEval_response_toxicity",
                              "UpTrain_Response_Completeness","UpTrain_Response_Conciseness","UpTrain_Response_Relevance",
                              "UpTrain_Response_Valid","UpTrain_Response_Consistency","UpTrain_Response_Response_Matching",
                              "UpTrain_Retriever_Context_Relevance","UpTrain_Retriever_Context_Utilization",
                              "UpTrain_Retriever_Factual_Accuracy","UpTrain_Retriever_Context_Conciseness",
                              "UpTrain_Retriever_Code_Hallucination",])
from evaluate import UptrainEvaluate
evalAgent = EvalModelAgent(cfg)
for question, expected_answer, golden_context, golden_context_ids in zip(qa_dataset['test_data']['question'], qa_dataset['test_data']['answers'], qa_dataset['test_data']['golden_sentences'], qa_dataset['test_data']['golden_ids']):
    response = query_engine.query(question)
    # 返回node节点
    retrieval_ids = []
    retrieval_context = []
    for source_node in response.source_nodes:
        retrieval_ids.append(source_node.metadata['id'])
        retrieval_context.append(source_node.get_content())
    actual_response = response.response
    eval_result = evaluating(question,response, actual_response, retrieval_context, retrieval_ids,
                             expected_answer, golden_context, golden_context_ids, evaluateResults.metrics,
                             evalAgent)
    evaluateResults.add(eval_result)
    all_num = all_num + 1
    evaluateResults.print_results()
    print("总数：" + str(all_num))


if __name__ == '__main__':
    print('Success')
