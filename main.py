from PyPDF2 import PdfReader
from dotenv import load_dotenv
import uuid
load_dotenv()
text=""
pdf_reader=PdfReader("Stock-Investing-101-eBook.pdf")
for page in pdf_reader.pages:
    text=text+page.extract_text() + "\n"

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(text)
# print(type(texts))
# print(texts[1])

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone()
index_name = "rag-demo"

# pc.create_index(
#     name=index_name,
#     dimension=768, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
# chunks_with_prefix = ["passage: " + chunk for chunk in texts]
# embeddings = model.encode(chunks_with_prefix, show_progress_bar=True, convert_to_numpy=True)
# # print(embeddings[0])
index = pc.Index(index_name)
# vectors = [{
#     "id": str(uuid.uuid4()),  # Unique ID for each chunk
#     "values": embedding.tolist(),  # Convert numpy array to list
#     "metadata": {
#         "text": texts[i]  # Optional: store the original text for later search/preview
#     }
# } for i, embedding in enumerate(embeddings)]

# # Step 3: Upsert to Pinecone
# index.upsert(vectors=vectors)

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
llm=ChatGroq(model="llama3-70b-8192")
while True:
    query=input("Ask a question (or type exit to quit)")
    if query.lower()=="exit":
        break
    query_embedding = model.encode(["passage: " + query], convert_to_numpy=True)[0]

    # 2. Query Pinecone for top 5 relevant chunks
    result = index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True
    )
    retrieved_texts = [match['metadata']['text'] for match in result['matches']]
    prompt_text="""
    You are an helpful assistant. Use the context provided to answer the user question. 
    
    Compare the Knowledge from Document and Your Own Internal Knowledge,

    Whichever is logically try, provide only that answer

    "Context": {context}
    "User Question": {question}
    """
    prompt=PromptTemplate(
        input_variables=["context","question"],
        template=prompt_text
    )
    chain=prompt | llm

    response=chain.invoke({"context":retrieved_texts,"question":query})
    print(f"Retrieved Text: {retrieved_texts}")
    print(f"Answer is: {response.content}")








