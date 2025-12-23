import os
from pinecone import Pinecone
from concurrent.futures import ThreadPoolExecutor

from google.genai import types
from google import genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

client = genai.Client()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("rag-notes")


def get_documents_from_path(path='./ko'):
    result = []
    for root, _, files in os.walk(path):
        for file in files:
            if not file.endswith('.md'):
                continue

            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            doc = Document(
                page_content=content,
                metadata={'name': file, 'path': file_path}
            )
            result.append(doc)
    return result


def embed_chunk(chunk_text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk_text,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    [embedding_obj] = result.embeddings
    return embedding_obj.values


def upsert_documents(documents, splitter, namespace=None):
    for document in documents:
        chunks = splitter.split_documents([document])
        total_chunks = len(chunks)

        vectors = []
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{namespace}_{document.metadata['path']}_{idx}"
            metadata = {
                'file_name': document.metadata['name'],
                'file_path': document.metadata['path'],
                'text': document.page_content,
                'chunk_index': idx,
                'total_chunks': total_chunks
            }
            vectors.append({'id': chunk_id, 'metadata': metadata,
                           'values': embed_chunk(chunk.page_content)})

        index.upsert(vectors=vectors, namespace=namespace)
        print(
            f"Upserted {len(vectors)} vectors to namespace '{namespace}' for document '{document.metadata['path']}'")


if __name__ == "__main__":
    docs = get_documents_from_path('./ko')

    tasks = [
        # 0% overlap
        (docs, 'cs256-ov0', RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)),
        (docs, 'cs512-ov0', RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)),
        (docs, 'cs1024-ov0',
         RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)),
        # 15% overlap
        (docs, 'cs256-ov15',
         RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=38)),
        (docs, 'cs512-ov15',
         RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=75)),
        (docs, 'cs1024-ov15',
         RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=154)),
        # 30% overlap
        (docs, 'cs256-ov30',
         RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=77)),
        (docs, 'cs512-ov30',
         RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=150)),
        (docs, 'cs1024-ov30',
         RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=307)),
    ]

    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = [executor.submit(upsert_documents, docs, splitter=sp, namespace=ns)
                   for docs, ns, sp in tasks]
        for future in futures:
            future.result()
