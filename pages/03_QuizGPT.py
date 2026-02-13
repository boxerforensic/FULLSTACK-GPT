import json
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser

class JsonOutPutParser(BaseOutputParser):
    
    def parse(self, text):
        #결과값에 불필요한 것을 제거해주고, json.loads()파일을 불러옴.
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutPutParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon = "?"
)

st.title("QuizGPT")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature = 0.1,
    streaming = True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def format_docs(docs): #documentGPT함수 재활용
    return "\n\n".join(doc.page_content for doc in docs)

questions_prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are a helpful assistant that is role playing as a teacher.
         
         Based Only on the following context make 10 question to test the user's knowledge about the text.
         
         Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
         Use (o) to signal the correct answer.
         
         Question examples:
          Question: What is the color of the ocean?
          Answers: Red| Yellow| Green| Blue(o) 
          
          Question: What is the capital or Georgia?
          Answers: Baku| Tbilisi(o)| Manila| Beirut
          
          Question: When was Avatar released?
          Answers: 2007|2001|2009(o)|1998
          
          Question: Who was Julius Caesar?
          Answers: A Roman Emperor(o)| Painter| Actor| Model
          
          your turn!
          
          Context:{context}
         """,)
        ]
    )

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are a helpful powerful formatting algorithm.

         You format exam question into JSON algorithm.
         Answer with(o) are the correct ones.
         
         Examples Input:
          Question: What is the color of the ocean?
          Answers: Red| Yellow| Green| Blue(o) 
          
          Question: What is the capital or Georgia?
          Answers: Baku| Tbilisi(o)| Manila| Beirut
          
          Question: When was Avatar released?
          Answers: 2007|2001|2009(o)|1998
          
          Question: Who was Julius Caesar?
          Answers: A Roman Emperor(o)| Painter| Actor| Model
          
          Example Output:
          
          ```json
          {{ "questions": [
              {{
                  "question":"What is the color of the ocean?",
                  "answers": [
                      {{
                          "answer":"red",
                          "correct":false
                      }},
                      {{
                          "answer":"Yellow",
                          "correct":false
                      }},
                      {{
                          "answer":"Green",
                          "correct":false
                      }},
                      {{
                          "answer":Blue",
                          "correct":True
                      }},
                  ]
                  
              }}
              {{
                  "question":"What is the capital of Georgia?",
                  "answers": [
                      {{
                          "answer":"Baku",
                          "correct":false
                      }},
                      {{
                          "answer":"Tbilisi",
                          "correct":true
                      }},
                      {{
                          "answer":"Manila",
                          "correct":false
                      }},
                      {{
                          "answer":Beirut",
                          "correct":false
                      }},
                  ]
                  
              }}
              {{
                  "question":"When was Avatar released?",
                  "answers": [
                      {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                  ]
                  
              }}
              {{
                  "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                  ]
                  
              }}
          ]
              
          }}
          ```
          
          your turn!
          
          Context:{context}
         """,)
        ]
    )

formatting_chain=formatting_prompt | llm

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

#cache_data가 해시를 할 수 없을 때는 topic이라는 변경 변수를 넣어서 다시 계산하도록 지시하는것, 없으면 다시 계산하지 않고 항상 전에 있ㄴ던 걸로 
@st.cache_data(show_spinner="Making Quiz...")
def run_quiz_chain(_docs, topic): 
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5) #상위 5개까지 문서를 전달함.
    docs = retriever.get_relevant_documents(term)
    return docs

with st.sidebar:
    docs = None
    choice = st.selectbox( #바로 붙여서 써야함. 
        "choose what you want to use.", 
     (
        "File", 
        "wikipedia Article",
    ),
     )
    
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf", 
            type=["pdf","txt","docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
            

if not docs:
    st.markdown("""
    Welcome to QuizGPT.
    
    I will make a quiz from Wikipedia articles of files you upload to test your knowledge and help your study.
    
    Get started by uploading a file of searching on Wikipedia in the sidebar.           
    """)

else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            #answers 안에 question["answers"]내부에 answer를 가지고 와서 그 안의 answer를 가지고 와서 표현해라.
            #index=None이면 아무것도 표시되지 않는것.
            value = st.radio("Select an option.", 
                     [answer["answer"] for answer in question["answers"]],
                     index=None) 
            #st.success는 성공을 st.error는 틀렸음을 나타냄
            if st.write({"answer": value, "correct": True} in question["answers"]):
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
            #json파일 안에, 답변과 true가 일치하는 쌍을 찾는데 있으면 맞은거고 없으면 틀림.
        button = st.form_submit_button()