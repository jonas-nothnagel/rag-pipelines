import pandas as pd
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import (
    TokenTextSplitter)

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

import os
import re
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'running on device: {device}')

if __name__ == '__main__':

    # import data
    markdown_files = []
    for root, dirs, files in os.walk("./data/processed_files"):
        for file in files:
            if file.lower().endswith('.md'):
                markdown_files.append(os.path.join(root, file))

    # Iterate over the file paths
    loaded_documents = []
    for doc in markdown_files:
        try:
            loader = UnstructuredMarkdownLoader(doc)
            documents = loader.load()
            loaded_documents.extend(documents)
            print(f"Loaded: {doc}")
        except Exception as e:
            print(f"Error loading {doc}: {str(e)}")

    # 2. Custom function to normalize text
    def normalize_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters (customize as needed)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    # Apply normalization to each document
    for doc in loaded_documents:
        doc.page_content = normalize_text(doc.page_content)

    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=150, length_function=len)
    docs = text_splitter.split_documents(loaded_documents)

    #load models
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-70B",
        task="text-generation",
        max_new_tokens=512,
        timeout=5,
        repetition_penalty=1.03,
    )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    generator_llm = llm
    critic_llm = llm
    embeddings = embeddings

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    # Change resulting question type distribution
    distributions = {  # uniform distribution
        simple: 0.1,
        reasoning: 0.35,
        multi_context: 0.2,
        conditional: 0.35
    }

    # use generator.generate_with_llamaindex_docs if you use llama-index as document loader
    testset = generator.generate_with_langchain_docs(docs, 50, distributions, raise_exceptions=False) 
    testset.to_pandas()
    
    #store data
    testset.to_parquet('synthetic_data/ragas_llama3_qa.parquet')

    print("Done!")