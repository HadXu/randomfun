import qdrant_client

"""
"""

key = "CQl_5MMOud-NZGPH4Bbd4mYDqtGhsH0XUUmGNYKWFRUjAQEFZ_45uw"

# client = qdrant_client.QdrantClient(":memory:", prefer_grpc=True)
client = qdrant_client.QdrantClient("https://7a908601-e33d-406b-b127-bd5340b2c215.us-east4-0.gcp.cloud.qdrant.io:6333", api_key=key)
x = client.get_collections()

print(x)

# client.add(
#     collection_name="knowledge-base",
#     documents=[
#         "Qdrant is a vector database & vector similarity search engine. It deploys as an API service providing search for the nearest high-dimensional vectors. With Qdrant, embeddings or neural network encoders can be turned into full-fledged applications for matching, searching, recommending, and much more!",
#         "Docker helps developers build, share, and run applications anywhere — without tedious environment configuration or management.",
#         "PyTorch is a machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing.",
#         "MySQL is an open-source relational database management system (RDBMS). A relational database organizes data into one or more data tables in which data may be related to each other; these relations help structure the data. SQL is a language that programmers use to create, modify and extract data from the relational database, as well as control user access to the database.",
#         "NGINX is a free, open-source, high-performance HTTP server and reverse proxy, as well as an IMAP/POP3 proxy server. NGINX is known for its high performance, stability, rich feature set, simple configuration, and low resource consumption.",
#         "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.",
#         "SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings. You can use this framework to compute sentence / text embeddings for more than 100 languages. These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning. This can be useful for semantic textual similar, semantic search, or paraphrase mining.",
#         "The cron command-line utility is a job scheduler on Unix-like operating systems. Users who set up and maintain software environments use cron to schedule jobs (commands or shell scripts), also known as cron jobs, to run periodically at fixed times, dates, or intervals.",
#     ]
# )

prompt = """
What tools should I need to use to build a web service using vector embeddings for search?
"""

import os
import openai

openai.api_base = 'https://proxy.openmao.icu/v1'
openai.api_key = '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'

# Fill the environmental variable with your own OpenAI API key
# See: https://platform.openai.com/account/api-keys
# os.environ["OPENAI_API_KEY"] = "<< PASS YOUR OWN KEY >>"

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[
        {"role": "user", "content": prompt},
    ]
)
print(completion["choices"][0]["message"]["content"])

results = client.query(
    collection_name="knowledge-base",
    query_text=prompt,
    limit=3,
)

context = "\n".join(r.document for r in results)
context

metaprompt = f"""
You are a software architect. 
Answer the following question using the provided context. 
If you can't find the answer, do not pretend you know it, but answer "I don't know".

Question: {prompt.strip()}

Context: 
{context.strip()}

Answer:
"""

# Look at the full metaprompt
print(metaprompt)

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": metaprompt},
    ],
    timeout=10.0,
)
print(completion["choices"][0]["message"]["content"])