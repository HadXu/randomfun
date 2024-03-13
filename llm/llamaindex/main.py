from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
    GPTVectorStoreIndex,
)
import openai
import os

openai.api_base = 'https://proxy.openmao.icu/v1'
openai.api_key = ''

PERSIST_DIR = "./storage"
# if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    # documents = SimpleDirectoryReader("data").load_data()
#     index = VectorStoreIndex.from_documents(documents)
#     # store it for later
#     index.storage_context.persist(persist_dir=PERSIST_DIR)
# else:
#     # load the existing index
#     storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
#     index = load_index_from_storage(storage_context)

# query_engine = index.as_query_engine()
# response = query_engine.query("什么是transformers？")
# print(response)

documents = SimpleDirectoryReader("data").load_data()
# print(documents)

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

key = ""
# client = qdrant_client.QdrantClient("https://7a908601-e33d-406b-b127-bd5340b2c215.us-east4-0.gcp.cloud.qdrant.io:6333", api_key=key)
vector_store = QdrantVectorStore(client=client, collection_name="NewsCategoryv3PoliticsSample")
service_context = ServiceContext.from_defaults(chunk_size_limit=512)
index = GPTVectorStoreIndex.from_documents(documents, vector_store=vector_store, service_context=service_context)

query_engine = index.as_query_engine(similarity_top_k=10)


response = query_engine.query("什么是transformers？")
print(response)
