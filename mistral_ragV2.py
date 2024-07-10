from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def configure_settings():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = None
    Settings.chunk_size = 256
    Settings.chunk_overlap = 25

def load_and_refine_documents(directory):
    documents = SimpleDirectoryReader(directory).load_data()
    refined_documents = [doc for doc in documents if all(
        phrase not in doc.text for phrase in ["Member-only story", "The Data Entrepreneurs", " min read"])]
    return refined_documents

def create_index(documents):
    return VectorStoreIndex.from_documents(documents)

def configure_retriever(index, top_k):
    return VectorIndexRetriever(index=index, similarity_top_k=top_k)

def create_query_engine(retriever):
    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )

def generate_context(response, top_k):
    return "Context:\n" + "\n\n".join(node.text for node in response.source_nodes[:top_k])

def load_model_and_tokenizer(model_name, peft_model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=False, revision="main"
    )
    config = PeftConfig.from_pretrained(peft_model_name)
    model = PeftModel.from_pretrained(model, peft_model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
    return tokenizer.batch_decode(outputs)[0]

def main():
    configure_settings()

    documents = load_and_refine_documents("articles")
    print(f"Number of documents after refinement: {len(documents)}")

    index = create_index(documents)
    top_k = 3
    retriever = configure_retriever(index, top_k)
    query_engine = create_query_engine(retriever)

    query = "What is fat-tailedness?"
    response = query_engine.query(query)
    context = generate_context(response, top_k)
    print(context)

    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    peft_model_name = "shawhin/shawgpt-ft"
    model, tokenizer = load_model_and_tokenizer(model_name, peft_model_name)

    instructions_string = "Please respond to the following comment."
    prompt_template = lambda comment: f"[INST] {instructions_string} \n{comment} \n[/INST]"
    prompt_template_w_context = lambda context, comment: f"""
        Please respond to the {comment}. 
        Use the context above if it is helpful.
        {context}
        [/INST]
    """

    comment = "What is fat-tailedness?"
    
    print("===================== Without RAG =========================")
    prompt = prompt_template(comment)
    print(prompt)
    model.eval()
    response_without_rag = generate_response(model, tokenizer, prompt)
    print(response_without_rag)

    print("===================== With RAG =========================")
    prompt = prompt_template_w_context(context, comment)
    response_with_rag = generate_response(model, tokenizer, prompt)
    print(response_with_rag)

if __name__ == "__main__":
    main()
