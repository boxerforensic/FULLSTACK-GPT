import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy()) #Windows 이벤트 루프 에러

from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import streamlit as st


llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.1
)

answer_prompt = ChatPromptTemplate.from_template(
    """
        Using Only the following context answer the user's question. If you can't just say you don't know, don't make anything up.
        then, give a score to the answer between 0 and 5.
        If the answer answers the user question the score should be high, else it should be low.
        
        Make sure to always include the answer's score even if it's 0.
        
        Context:{context}
        
        examples:
        
        Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    
    Question: {question}
    
    """
)

def get_answer(input):
    docs = input["docs"]
    question = input["question"]
    answer_chain = answer_prompt | llm
    # answers = []
    # for doc in docs: #특정 기법 같은데, 파일들을 쪼개서 각각 질문하고 거기에 대해서 점수를 부여해서 제일 좋은 것으로 나타낸다.
    #     result = answer_chain.invoke({
    #         "question":question,
    #         "context": doc.page_content
    #     }){
    #     answers.append(result.content) #메타데이터는 넣지 않는다.
    # st.write(answers)
    return {
        "question": question,
        "answers":[ #lCEL에서는 결과물이 dic이어야만 한다.
        { "answer":answer_chain.invoke({
            "question":question, "context":doc.page_content}).content,
         "source": doc.metadata["source"],
         "date": doc.metadata["lastmod"]
        } for doc in docs #세가지 데이터를 dic형태로
    ]}

choose_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     Use Only the following pre-existing answers to answer the user's question.
     
     Use the answer that have the highest score (more helpful) and favor most favor the most recent ones.
     
     Return the sources of the answers as they are, do not change them.
     
     Answers: {answers}
     """,),
    ("human", "{question}"),
])

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n" for answer in answers)
    return choose_chain.invoke(
        {"question": question,
         "answers":condensed}
    )
 
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose() #전체페이지에서 header제거
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("★", "")
        .replace("\n\n\n\nesbuild can build css\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nSkip to main content\n\n\n\n\nFavorites\nTIL\nZines\nRSS\n\n\n\n\n\n\n\n\n", "")
        ) #header와 footer를 제거한 soup을 반환
    

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
    loader = SitemapLoader(
        url,
        # filter_urls=[
        #     r"^(?!.*\/blog\/).*",
        #     ],
        parsing_function=parse_page
        )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        raise ValueError("로드된 문서가 0개입니다. (요청 차단/429/파싱 결과 빈 텍스트 가능)")
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = {"docs":retriever, "question":RunnablePassthrough()} | RunnableLambda(get_answer) | RunnableLambda(choose_answer)
        
            result = chain.invoke(query)
            st.write(result.content)   