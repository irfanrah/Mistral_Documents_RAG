from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25



documents = SimpleDirectoryReader("articles").load_data()


# some ad hoc document refinement
print(len(documents))
for doc in documents:
    if "Member-only story" in doc.text:
        documents.remove(doc)
        continue

    if "The Data Entrepreneurs" in doc.text:
        documents.remove(doc)

    if " min read" in doc.text:
        documents.remove(doc)

print(len(documents))


# store docs into vector DB
index = VectorStoreIndex.from_documents(documents)


# set number of docs to retreive
top_k = 3

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)


# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

# query documents
query = "What is fat-tailedness?"
response = query_engine.query(query)


# reformat response
context = "Context:\n"
for i in range(top_k):
    context = context + response.source_nodes[i].text + "\n\n"

print(context)


# load fine-tuned model from hub
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

config = PeftConfig.from_pretrained("shawhin/shawgpt-ft")
model = PeftModel.from_pretrained(model, "shawhin/shawgpt-ft")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


# prompt (no context)
intstructions_string = f"""
                        Please respond to the following comment.
                        """

prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''


comment = "What is fat-tailedness?"
print("===================== Without RAG =========================")
prompt = prompt_template(comment)
print(prompt)


model.eval()

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

print(tokenizer.batch_decode(outputs)[0])


print("===================== With RAG =========================")
prompt_template_w_context = lambda context, comment: f"""   
                                                    Please respond to the {comment}. 
                                                    
                                                    Use the context above if it is helpful.

                                                    {context}
                                                    
                                                    [/INST]
                                                    """


prompt = prompt_template_w_context(context, comment)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

print(tokenizer.batch_decode(outputs)[0])