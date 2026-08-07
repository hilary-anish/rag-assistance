import streamlit as st
import httpx

st.set_page_config(page_title="Document-Intelligence RAG", page_icon="🔍", layout="centered")

st.title("Document-Intelligence RAG Assistant")
st.caption("Ask questions about EU regulations, research papers, and technical documentation.")

api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating…"):
            try:
                resp = httpx.post(
                    f"{api_url.rstrip('/')}/ask",
                    json={"question": prompt, "k": 5},
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.ConnectError:
                st.error(f"Cannot connect to the API at {api_url}. Is the server running?")
                st.stop()
            except httpx.HTTPStatusError as exc:
                st.error(f"API error: {exc.response.status_code} — {exc.response.text[:200]}")
                st.stop()
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

        conf = data["confidence"]
        flag = data["flag"]
        if flag == "grounded":
            badge = f":green[✅ Grounded — {conf:.0%} confidence]"
        else:
            badge = f":orange[⚠️ Low Confidence — {conf:.0%} confidence]"
        st.markdown(badge)

        st.markdown(data["answer"])

        with st.expander("Sources"):
            for i, src in enumerate(data["sources"], 1):
                st.markdown(f"**[{i}]** `{src['source']}` · page {src['page']}")
                st.text(src["text"])

        reply_md = f"{badge}\n\n{data['answer']}"
        st.session_state.messages.append({"role": "assistant", "content": reply_md})
