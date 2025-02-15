import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーの読み込み
model_path = "Rakuten/RakutenAI-2.0-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# GPUが利用可能ならGPUを使用、なければCPUを使用
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map={"": device})
model.eval()

# Streamlitアプリの設定
st.title("RakutenAI-2.0 Chatbot")
st.write("RakutenAI-2.0-mini-instructモデルを使用したチャットボットです。")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "あなたは親切で丁寧なAIアシスタントです。"}
    ]

# ユーザー入力欄
user_input = st.text_input("あなたのメッセージを入力してください：", key="user_input")

# 送信ボタン
send_clicked = st.button("送信")

if send_clicked and user_input:
    # ユーザーのメッセージをチャット履歴に追加
    st.session_state.messages.append({"role": "user", "content": user_input})

    # モデルへの入力を準備
    input_ids = tokenizer.apply_chat_template(
        st.session_state.messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device=model.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    # モデルによる応答生成
    with st.spinner("AIが応答を生成しています..."):
        tokens = model.generate(
            input_ids,
            max_length=2048,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )
    assistant_message = tokenizer.decode(tokens[0][len(input_ids[0]):], skip_special_tokens=True)

    # AIの応答をチャット履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    # **ユーザー入力をリセット**
    del st.session_state["user_input"]

# チャット履歴の表示
if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"**ユーザー**: {message['content']}")
        elif message["role"] == "assistant":
            st.write(f"**AIアシスタント**: {message['content']}")
