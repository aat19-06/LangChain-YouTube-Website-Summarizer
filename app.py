import validators,streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


## Streamlit APP
st.set_page_config(page_title="Langchain: Summarize Text From YT or Website",page_icon="🌐")
st.title("Langchain: Summarize Text From YT or Website")
st.subheader('Summarize URL')



# GROQ API KEY and url (YT OR WEBSITE)
with st.sidebar:
    groq_api_key=st.text_input("Groq API KEY",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

llm=ChatGroq(groq_api_key=groq_api_key,model="llama-3.3-70b-versatile")

prompt_template="""
Provide a summary of the  following content in 300 words
Content:{text}"""

prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provise the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url . It can may be a YT video url or website url ")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                ## Chain for summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                summary=chain.run(docs)

                st.success(summary)

        except Exception as e:
            st.exception(f"Exception:{e}")

                    



