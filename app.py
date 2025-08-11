import os
import streamlit as st
#from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.llms import Ollama
load_dotenv()


#langsmith tracking
os.environ["LANGCHAIN_APT_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")


#prompt templet
prompt=ChatPromptTemplate(
    [
        ("system","you are a helpful assistant. Please response to the user queries"),
        ("user","question:{question}")
    ]
)

# def generate_response(question, api_key, model, temperature, max_tokens):
#     try:
#         llm = ChatOpenAI(
#             model=model,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             openai_api_key=api_key
#         )
#         output_parser = StrOutputParser()
#         chain = prompt | llm | output_parser
#         return chain.invoke({"question": question})
#     except Exception as e:
#         if "insufficient_quota" in str(e):
#             return " You have exceeded your OpenAI quota. Please check your billing details."
#         return f"Error: {str(e)}"



def generate_response(question,temperature,max_tokens):
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=temperature,
        max_new_tokens=max_tokens
    )
    model=ChatHuggingFace(llm=llm)
    output_parser=StrOutputParser()
    chain=prompt|model|output_parser
    return chain.invoke({"question":question})

# def generate(question,tempearture,max_tokens,engine):
#     llm=Ollama(model=engine) 
#     output=StrOutputParser()
#     chain=prompt|llm|output
#     answer=chain.invoke({"question":question})
#     return answer


# engine=st.sidebar.selectbox("select model",["mistral"])


#title of the app
st.title("Q&A chatbot with OpenAI")

#sidebar for settings
st.sidebar.title("settings")
#api_key=st.sidebar.text_input("enter your Open AI API key:",type="password")

#drop down to select the open ai model
# model = st.sidebar.selectbox(
#     "Select an OpenAI Model",
#     ["gpt-5", "gpt-4o"]
# )

#tempeture value
temperature=st.sidebar.slider("temperature",min_value=0.0,max_value=1.0,value=0.7)
#maxtokens
max_tokens=st.sidebar.slider("max_tokens",min_value=50,max_value=300,value=150)


#main interface
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,temperature,max_tokens)
    st.write(response)
else:
    st.write("please provide the query")