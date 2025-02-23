from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType,
)
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.document_loaders.sitemap import SitemapLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate

from pypdf import PdfReader
from tqdm import tqdm

# PDFデータ読み込み
def read_pdf_data(pdf_file):
    pdf_page = PdfReader(pdf_file)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text


# 読み込んだテキストをチャンク単位で小分け
def split_data(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
    )
    
    if isinstance(text, str):
        docs = text_splitter.split_text(text) # テキストをチャンクに分割
        docs_chunks = text_splitter.create_documents(docs) # チャンクをドキュメントに変換
    else:
        docs_chunks = text_splitter.split_documents(text) # ドキュメントをチャンクに分割
    return docs_chunks


# FAISSにベクトルデータを保存
# def add_to_faiss(faiss_db, docs, embeddings):

#     with tqdm(total=len(docs), desc="documents ベクトル化") as pbar:
#         for d in docs:
#             if faiss_db:
#                 faiss_db.add_documents([d])
#             else:
#                 faiss_db = FAISS.from_documents([d], embeddings)
#             pbar.update(1)
#     return faiss_db

# MongoDBにベクトルデータを保存
def add_to_cosmos(cosmos_db, docs, embeddings, collection, index_name):
  with tqdm(total=len(docs), desc="documents ベクトル化") as pbar:
    for d in docs:
      if cosmos_db :
        cosmos_db.add_documents([d])
      else:
        cosmos_db = AzureCosmosDBVectorSearch.from_documents(
          [d],
          embeddings,
          collection = collection,
          index_name = index_name
        )
      pbar.update(1)

  num_documents = collection.count_documents({})
  num_lists = 1 if num_documents < 100000 else num_documents // 1000 # IVFクラスターの数、documentsの数/1000
  dimensions = 1536
  similarity_algorithm = CosmosDBSimilarityType.COS # cos類似度
  kind = CosmosDBVectorSearchType.VECTOR_IVF # 検索するドキュメントをクラスターで分割
  m = 16
  ef_construction = 64
  ef_search = 40
  score_threshold = 0.1

  cosmos_db.create_index(
      num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
  )
  return cosmos_db

def pull_from_cosmos(embeddings, conn_str, name_space, index_name):
   vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
      conn_str, name_space, embeddings, index_name=index_name
   )
   retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # 上位k件を取得、他にも検索のオプションを設定可能
   return retriever


def get_contextualize_prompt_chain(model):
  contextualize_q_system_prompt = (
    "あなたは、AIでチャットの質問を作り直すように求められています。"
    "チャット履歴と最新のユーザーメッセージがあり、そのメッセージは"
    "チャット履歴のコンテキストを参照している質問である可能性があります。"
    "チャット履歴がなくても、理解できる独立した質問を作成してください。"
    "絶対に、質問に答えないでください。"
    "質問は、「教えてください。」「どういうことですか？」などAIに投げかける質問にしてください。"
    "メッセージが質問であれば、作り直してください。"
    "「ありがとう」などメッセージが質問ではない場合は、メッセージを作り直さず戻してください。"
    "\n\n"
  )
  contextualize_q_prompt = ChatPromptTemplate.from_messages(  
    [
      ("system", contextualize_q_system_prompt),
      MessagesPlaceholder("chat_history"),
      ("human", "{input}")
    ]
  )
  contextualize_chain = contextualize_q_prompt | model | StrOutputParser()

  return contextualize_chain
  

def get_chain(model):
  system_prompt = (
    "あなたは質問対応のアシスタントです。"
    "質問に答えるために、検索された文脈の以下の部分を使用してください。"
    "答えがわからない場合は、わからないと答えてください。"
    "回答は3文以内で簡潔にしてください。"
    "\n\n"
    "{context}"
  )

  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system_prompt),
      MessagesPlaceholder("chat_history"),
      ("human", "{input}")
    ]
  )

  chain = prompt | model 
  return chain

# def pull_from_faiss(embeddings, faiss_db_dir="vector_store"):
#   vectorstore = FAISS.load_local(
#     faiss_db_dir,
#     embeddings,
#     allow_dangerous_deserialization=True#ファイル読み込み時のセキュリティチェックが緩和される
#   )
#   retriver = vectorstore.as_retriever()
#   return retriver


