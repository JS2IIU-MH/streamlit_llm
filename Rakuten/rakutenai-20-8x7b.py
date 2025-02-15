from transformers import AutoModelForCausalLM, AutoTokenizer

import streamlit as st

# lzmaのチェック
try:
    import lzma
except ImportError:
    st.error("Pythonにlzmaモジュールがありません。liblzmaをインストールしてください。")
    st.stop()


model_path = "Rakuten/RakutenAI-2.0-8x7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
model.eval()


# セッションステートの初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# メッセージ履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの入力を受け取る
if prompt := st.chat_input("メッセージを入力してください..."):
    # ユーザーのメッセージを履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    # チャットメッセージとして表示
    with st.chat_message("user"):
        st.markdown(prompt)

    # モデルからの応答を生成
    input_text = tokenizer(prompt, return_tensors="pt").to(device=model.device)
    tokens = model.generate(
        **input_text,
        max_new_tokens=512,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(tokens[0], skip_special_tokens=True)

    # モデルのメッセージを履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": response})
    # チャットメッセージとして表示
    with st.chat_message("assistant"):
        st.markdown(response)
