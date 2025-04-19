

import argparse
import json
import time
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Efficient database for similarity search using FAISS.
from langchain_community.vectorstores import FAISS


# Embeddings model for generating document embeddings. To convert text to vectors.
from langchain_community.embeddings import HuggingFaceEmbeddings

# Prompt template for summarization, Helps to define the prompt structure for the language model.
from langchain.prompts import PromptTemplate

# Document loader to load text data from a file or a list.
from langchain_community.document_loaders import TextLoader

# Document class to wrap text data in to structured document format for processing.
from langchain.schema import Document



# Disable all the warnings those are occuring in the terminal.

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load embeddings model to create the data set for the FAISS vectorstore.

# The model used here is "sentence-transformers/all-MiniLM-L6-v2" which is a small, fast, and efficient model for sentence embeddings.

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the data set from the json file.

def build_vectorstore(corpus):
    """
    Build a FAISS vectorstore from a list of documents.
    """
# Wrap corpus in LangChain Document objects

    documents = [Document(page_content=doc) for doc in corpus]
    
# Generate embeddings and build FAISS index from documents  
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore


# Argument parser to handle input parameters

if __name__ == "__main__":
    

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--save_fp', type=str, default='results/gpt4_summaries.json')
    argparser.add_argument('--dataset_fp', type=str, default='data/billsum_train.json')
    argparser.add_argument('--key', type=str, required=True)
    argparser.add_argument('--model', type=str, default='gpt-4-0613')
    args = argparser.parse_args()


# Seting up the OpenAI API key while running the python file provide the API key as an argument.

# Note - :   type in terminal python RAG.py --key <your_openai_api_key>

# Example: python RAG.py --key sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


    openai_api_key = args.key

# Load dataset

    dataset = json.load(open(args.dataset_fp))



# Sample corpus of documents (replace this with the actual documents in your use case) 

# Here corpus will be the summaries 
    corpus = ["Document 1 content...", "Document 2 content...", "Document 3 content..."]


    
# Build FAISS vectorstore from the corpus documents using HuggingFace embeddings.

# This function will generate embeddings for each document and build a FAISS index for efficient similarity search.
    
    vectorstore = build_vectorstore(corpus)

# Initialize LangChain components for retrieval-augmented generation

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(model=args.model, openai_api_key=openai_api_key, temperature=0.7)

# Define prompt template for summarization

    prompt_template = """
    You are a helpful assistant. Summarize the following text:

    Document: {text}

    Use the following relevant documents for context:
    {context}

    Summary:"""
    prompt = PromptTemplate(input_variables=["text", "context"], template=prompt_template)


    # Create a retrieval-augmented QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Directly pass documents as context
        retriever=retriever,
        return_source_documents=False
    )
# Generate summaries("model output") for each dataset entry and store results in a new JSON file
    #  

    ct, ignore = 0, 0
    new_json = []


    for instance in dataset:
        try:
            # Extract text and summary from the dataset entry
            text = instance['text']
            original_summary = instance['summary']
            
            # Retrieve relevant documents from the vectorstore
            context_docs = retriever.get_relevant_documents(text)
            context = "\n".join([doc.page_content for doc in context_docs])

            # Format the prompt with the text and context
            formatted_prompt = prompt.format(text=text, context=context)

            # Get the model's response (summary) for the formatted prompt
            model_response = qa_chain.run(formatted_prompt)
           
            # Store the result
            result_entry = {
                "text": text,
                "summary": original_summary,  # Original summary from dataset
                "model_response": model_response,  # Response from model
            }
            new_json.append(result_entry)
            ct += 1
        except Exception as e:
            print(f"Error processing entry: {e}")
            ignore += 1
            continue

# Saving all the new JSON with model response to the specified file path

    with open(args.save_fp, "w") as f:
        json.dump(new_json, f, indent=4)

    print(f"Summaries saved to {args.save_fp}")
    print(f"Processed: {ct}, Ignored: {ignore}")

