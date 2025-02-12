import streamlit as st
from src.utils import *
import os 
import sys
from dotenv import load_dotenv

from langchain_openai import (
  AzureOpenAIEmbeddings,
  OpenAIEmbeddings,
  AzureChatOpenAI,
  ChatOpenAI
)

from langchain_core.messages import (
  HumanMessage,
  AIMessage
)

USER_NAME = "user"
ASSISTANT_NAME = "assistant"

def main() :

  st.title('Azure Custom Chat')

  # クリアボタン
  with st.sidebar:
     if st.button("Clear Chat"):
        st.session_state.chat_log = []
     message_num = st.slider("会話履歴数", min_value=5, max_value=50, value=10)

  #環境変数の取得
  openAI_key = os.getenv('AZURE_OPENAI_API_KEY')
  openAI_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

  embeddings = None
  if openAI_key != "" :
     embeddings = AzureOpenAIEmbeddings(
        api_key=openAI_key,
        azure_deployment="text-embedding-ada-002",
        # openai_api_versiton="2024-10-21",
        azure_endpoint=openAI_endpoint
     )
  else:
     st.error("EmbeddingのAPIKeyを設定してください")

  model = None
  if openAI_key != "":
     model = AzureChatOpenAI(
        api_key=openAI_key,
        azure_deployment="gpt-4",
        openai_api_version="2024-10-21",
        azure_endpoint=openAI_endpoint
     )   
  else:
     st.error("ChatモデルのAPIKeyを設定してください")

  #inputTextを変換するChainを取得
  contextualize_chain = get_contextualize_prompt_chain(model)
  #NewTextをcontextとして使用し、変換するChainを取得
  chain = get_chain(model)


  #FAISSからretrieverを取得
  if os.path.isdir('vector_store') :
    retriever = pull_from_faiss(embeddings)
  else :
     st.write("データをアップロードしてください。")
     return

  # チャットログを保存したセッション情報を初期化
  if "chat_log" not in st.session_state:
      st.session_state.chat_log = []

  user_msg = st.chat_input("ここにメッセージを入力")

  if user_msg:
    
    # 以前のチャットログを表示
    for chat in st.session_state.chat_log:
        if isinstance(chat, AIMessage):
          with st.chat_message(ASSISTANT_NAME):
              st.write(chat.content)
        else:
          with st.chat_message(USER_NAME):
              st.write(chat.content)


    # ユーザーのメッセージを表示
    with st.chat_message(USER_NAME):
      st.write(user_msg)

    # 今までの会話履歴を元に入力テキストを変換
    if st.session_state.chat_log:
      new_msg = contextualize_chain.invoke({"chat_history": st.session_state.chat_log, "input": user_msg})
    else:
      new_msg = user_msg
    print(user_msg, "=>", new_msg)


    #類似ドキュメントを取得
    relavant_docs = retriever.invoke(new_msg, k=3)

    #質問の回答を表示
    response = ""
    with st.chat_message(ASSISTANT_NAME):
      msg_placeholder = st.empty()

      for r in chain.stream({"chat_history": st.session_state.chat_log, "context": relavant_docs, "input": user_msg}):
        response += r.content
        msg_placeholder.markdown(response + "◼️")
      msg_placeholder.markdown(response)
      
      # ドキュメントのソースを表示
      col = st.columns(len(relavant_docs))
      for i, doc in enumerate(relavant_docs):
         with col[i]:
            with st.popover(f"ref{i+1}"):
               st.markdown(doc.page_content)

    # セッションにチャットログを追加
    st.session_state.chat_log.extend([
      HumanMessage(content=user_msg),
      AIMessage(content=response)
     ])
    
    # チャット履歴数を制限
    if len(st.session_state.chat_log) > message_num:
       st.session_state.chat_log = st.session_state.chat_log[-message_num:]
    

if __name__ == '__main__':
    load_dotenv('./../.env')
    main()

