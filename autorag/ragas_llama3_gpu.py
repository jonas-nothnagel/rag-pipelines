import pandas as pd
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from autorag.data.qacreation.ragas import generate_qa_ragas
import getpass
import os
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'running on device: {device}')


if __name__ == '__main__':

    # load data - need to use linux filepath syntax for running on Compute Cluster
    corpus_df = pd.read_parquet("synthetic_data/corpus.parquet", engine='pyarrow')

    # get ENV HF Token:
    from dotenv import load_dotenv
    load_dotenv()

    if "HF_API_KEY" not in os.environ:
        os.environ["HF_API_KEY"] = getpass("Enter HF API key:")

    #load models
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-70B",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    #run ragas
    distributions = {  # uniform distribution
    simple: 0.1,
    reasoning: 0.35,
    multi_context: 0.2,
    conditional: 0.35
    }

    qa_df = generate_qa_ragas(corpus_df, test_size=50, 
                          generator_llm=llm, 
                          critic_llm=llm, 
                          embedding_model=embeddings, 
                          distributions=distributions)
    
    #store data
    qa_df.to_parquet('synthetic_data/ragas_llama3_qa.parquet')

    print("Done!")