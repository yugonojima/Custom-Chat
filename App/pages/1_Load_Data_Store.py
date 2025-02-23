import os
import streamlit as st
from src.utils import *
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import pymongo


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

    # Vector Storeの初期化
    cosmos_db = None
    CONNECTION_STRING = os.environ.get("CONNECTION_STRING")
    INDEX_NAME = os.environ.get("INDEX_NAME")
    NAMESPACE = os.environ.get("NAMESPACE")
    DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

    # DB初期化のチェックボックス
    checkbox = st.checkbox("DB初期化")

    # アップロード pdf ファイル...
    st.session_state.uploaded_files = st.file_uploader("PDFファイル", type=["pdf"], accept_multiple_files=True)

    if st.button("アップロード"):
        # ファイルの数だけ処理を行う
        for i, pdf in enumerate(st.session_state.uploaded_files):
            with st.spinner(f'{i+1}/{len(st.session_state.uploaded_files)} 処理中...'):
                # 1. PDFデータ読み込み
                text=read_pdf_data(pdf)
                st.write("1. PDFデータ読み込み")

                # 2. データをチャンクに小分けにする
                docs_chunks=split_data(text)
                st.write("2. データをチャンクに小分けにする")

                # 3. CosmosDbにベクトル化して格納
                try:
                    client = pymongo.MongoClient(CONNECTION_STRING)
                    response = client.admin.command("ping")
                    if response.get("ok") == 1.0:
                        collection = client[DB_NAME][COLLECTION_NAME] #辞書のようにも振る舞う
                        if checkbox:
                            # collectionの初期化
                            collection.drop()
                            # indexを初期化
                            collection.drop_index(INDEX_NAME)
                        cosmos_db = add_to_cosmos(
                            cosmos_db = cosmos_db,
                            docs = docs_chunks,
                            embeddings = embeddings,
                            collection = collection,
                            index_name = INDEX_NAME
                        )
                        if cosmos_db is not None:
                            st.write("3. ベクトルデータの保存")
                            st.success("完了!")
                except Exception as e:
                    print(e)
                    st.write("3. ベクトルデータの保存失敗")
                    st.error("失敗!")



if __name__ == '__main__':
    main()
