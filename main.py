import streamlit as st
from rag import process_urls, generate_answer

st.title("Real Estate Research Tool")

url1 = st.sidebar.text_input("URL 1")
ur12 = st.sidebar.text_input("URL 2")
ur13 = st.sidebar.text_input("URL 3")
process_url_button = st.sidebar.button('Process URLs')

placeholder = st.empty()

if process_url_button:
    urls = [url for url in (url1, ur12, ur13) if url !='']
    print(urls)
    if len(urls) == 0:
        placeholder.text("you must provide atleast one url")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

query = placeholder.text_input("ask your question")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer")
        st.write(answer)
        if sources:
            st.subheader("Sources")
            for sources in sources.split("\n"):
                st.write(sources)
    except RuntimeError as e:
        placeholder.text("you must process th url first")
