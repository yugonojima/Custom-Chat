import os
import streamlit as st
from src.utils import *
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings


# langchain document_loaders参照
# https://python.langchain.com/v0.2/docs/integrations/document_loaders/

def main():
    st.title("PDFファイルをアップロード📁 ")

    openAI_key = os.getenv('AZURE_OPENAI_API_KEY')
    openAI_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

    # emmbeddingsのモデルを取得
    embeddings = None
    if openAI_key != "":
        # Azureの場合
        embeddings = AzureOpenAIEmbeddings(
        api_key=openAI_key,
        azure_deployment="text-embedding-ada-002",
        # openai_api_versiton="2024-10-21",
        azure_endpoint=openAI_endpoint
        )
    else:
        st.error("EmbeddingのAPIKeyを設定してください")

    # FAISSの初期化
    faiss_db = None
    st.write("0. FAISSの初期化")

    # アップロード pdf ファイル...
    uploaded_files = st.file_uploader("PDFファイル", type=["pdf"], accept_multiple_files=True)

    # ファイルの数だけ処理を行う
    for i, pdf in enumerate(uploaded_files):
        with st.spinner(f'{i+1}/{len(uploaded_files)} 処理中...'):
            # 1. PDFデータ読み込み
            text=read_pdf_data(pdf)
            st.write("1. PDFデータ読み込み")

            # 2. データをチャンクに小分けにする
            docs_chunks=split_data(text)
            st.write("2. データをチャンクに小分けにする")

            # 3. FAISSにベクトル化して格納
            faiss_db = add_to_faiss(
                faiss_db=faiss_db,               
                docs=docs_chunks,
                embeddings=embeddings
            )

    if faiss_db is not None:
        faiss_db.save_local("vector_store")
        st.write("3. ベクトルデータの保存")
        st.success("完了！")

if __name__ == '__main__':
    main()
