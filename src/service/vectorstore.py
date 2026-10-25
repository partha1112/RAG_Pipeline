from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import CrossEncoder
from pinecone import Pinecone

from src.llm import embeddings

load_dotenv()

pc = Pinecone()

index_name = "employee-rag-pipeline"

headers_vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace="headers",
)

chunks_vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace="chunks",
)

cache_vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace="cache",
)

bm25_encoder = BM25Encoder()
bm25_encoder.load("bm25_params.json")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_headers(input_query: str) -> dict:
    header_results = headers_vectorstore.similarity_search_with_score(
        input_query,
        k=1,
    )

    parent_header = None
    child_header = None
    sub_child_header = None

    if header_results and header_results[0][0].metadata:
        metadata = header_results[0][0].metadata
        parent_header = metadata.get("parent_header")
        child_header = metadata.get("child_header")
        sub_child_header = metadata.get("sub_child_header")

    return {
        "parent_header": parent_header,
        "child_header": child_header,
        "sub_child_header": sub_child_header,
    }


def get_documents(user_query: str) -> list:
    headers = get_headers(user_query)
    filter_conditions = []

    if headers.get("parent_header"):
        filter_conditions.append({
            "parent_header": {"$eq": headers["parent_header"]}
        })

    if headers.get("child_header"):
        filter_conditions.append({
            "child_header": {"$eq": headers["child_header"]}
        })

    if headers.get("sub_child_header"):
        filter_conditions.append({
            "sub_child_header": {"$eq": headers["sub_child_header"]}
        })

    if len(filter_conditions) > 1:
        pinecone_filter = {"$and": filter_conditions}
    elif len(filter_conditions) == 1:
        pinecone_filter = filter_conditions[0]
    else:
        pinecone_filter = None

    print(f"Applied Pinecone filter: {pinecone_filter}")

    chunk_results = chunks_vectorstore.similarity_search_with_score(
        query=user_query,
        k=20,
        filter=pinecone_filter,
    )

    return bm25_and_rerank_docs(user_query, chunk_results)


def bm25_and_rerank_docs( user_query: str, semantic_data: list ) -> list:
    if not semantic_data:
        return []

    query_sparse = bm25_encoder.encode_queries(user_query)

    if isinstance(query_sparse, list):
        query_sparse = query_sparse[0] if query_sparse else {}

    query_dict = dict(
        zip(
            query_sparse.get("indices", []),
            query_sparse.get("values", []),
        )
    )

    hybrid_scored_docs = []

    for doc, vector_score in semantic_data:
        doc_sparse = bm25_encoder.encode_documents(doc.page_content)

        if isinstance(doc_sparse, list):
            doc_sparse = doc_sparse[0] if doc_sparse else {}

        doc_dict = dict(
            zip(
                doc_sparse.get("indices", []),
                doc_sparse.get("values", []),
            )
        )

        bm25_score = sum(
            query_dict.get(token_id, 0.0) * weight
            for token_id, weight in doc_dict.items()
        )

        hybrid_score = vector_score + bm25_score
        hybrid_scored_docs.append((doc, hybrid_score))

    hybrid_scored_docs.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    query_doc_pairs = [
        [user_query, doc.page_content]
        for doc, _score in hybrid_scored_docs
    ]

    rerank_scores = reranker.predict(query_doc_pairs)

    scored_docs = list(
        zip(
            [doc for doc, _score in hybrid_scored_docs],
            rerank_scores,
        )
    )

    scored_docs.sort(
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return scored_docs[:3]



def get_documents_without_metadata(user_query: str):
    candidate_k = 5

    chunk_results = chunks_vectorstore.similarity_search_with_score(
        query=user_query,
        k=candidate_k
    )

    final_data = bm25_and_rerank_docs(
        user_query,
        chunk_results
    )

    return final_data

def add_to_cache(rewritten_input : str, res_str : str ):
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=768,  
            metric="cache",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    meta = {"llm_response": res_str}

    chunk = [ Document(page_content=rewritten_input, metadata=meta) ]

    vector_store = PineconeVectorStore.from_documents(
    documents=chunk,
    embedding=embeddings,
    index_name=index_name,
    namespace="cache",
    )

    print(vector_store)

def fetch_from_cache(user_query: str):
    cache_results = cache_vectorstore.similarity_search_with_score(
            query=user_query,
            k=1
        )
    print(cache_results)
    if cache_results and len(cache_results)>0:
        document, score = cache_results[0]
        if score >= 0.95:
            return document.metadata["llm_response"]
    return None