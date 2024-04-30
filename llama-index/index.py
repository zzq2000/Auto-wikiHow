from llama_index.core import VectorStoreIndex
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from loader import get_documents
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.node_parser import HierarchicalNodeParser

def get_index(sources, persist_dir, split_type="sentence", chunk_size=1024):
    if not os.path.exists(persist_dir):
        # load the documents and create the index
        documents = get_documents(sources)
        if split_type == "sentence":
            parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            index = VectorStoreIndex(nodes,show_progress=True)
        elif split_type == "character":
            parser = LangchainNodeParser(RecursiveCharacterTextSplitter())
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            index = VectorStoreIndex(nodes,show_progress=True)
        elif split_type == "hierarchical":
            parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128]
            )
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            index = VectorStoreIndex(nodes,show_progress=True)
        else:
            raise ValueError(f"split_type {split_type} not supported.")
        # store it for later
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    return index
