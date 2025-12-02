# La herramienta para construir la app web
import streamlit as st 
# La herramienta para la base de datos Chroma
from langchain_chroma import Chroma 
# La herramienta para el modelo de embeddings de OpenAI
from langchain_openai import OpenAIEmbeddings 
# La herramienta para el modelo de chat de OpenAI
from langchain_openai import ChatOpenAI 
# Las herramientas para construir la cadena RAG
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

DB_PATH = "db/chroma/" #para definir la ruta de la base de datos
EMBEDDING_MODEL = "text-embedding-3-small" #para definir el modelo de embeddings
LLM_MODEL = "gpt-3.5-turbo" #para definir el modelo de lenguaje


def format_docs(docs): #función para formatear los documentos recuperados sirve para concatenar los contenidos de los documentos recuperados
        return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def cargar_motor_rag():
    embeddings = OpenAIEmbeddings(model = EMBEDDING_MODEL) #para definir el modelo de embeddings
    
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever() #para definir el recuperador de documentos
    llm = ChatOpenAI(temperature=0, model_name=LLM_MODEL) #para definir el modelo de lenguaje
    parser = StrOutputParser() 
    
    prompt = ChatPromptTemplate.from_template(
        """
        Responde basandote en este contexto del nivel B2 de inglés: {context}
        Si no sabes la respuesta, di que no lo sabes.
        Pregunta = {question}
        Respuesta:
        """
    )
    
    rag_chain = (
    {
        "context": retriever | format_docs,  # Busca 
        "question": RunnablePassthrough()     # Pasa pregunta
    }
    | prompt  
    | llm     
    | parser  
    )
    
    return rag_chain
    
    
def main():
    st.markdown("🦜🔗 Chatbot Tutor de :red[Inglés Nivel B2]")
    st.subheader("Creado con arquitectura RAG usando LangChain, Chroma y OpenAI...")
    st.write("¡Hazme cualquier pregunta sobre vocabulario, gramática o cualquier tema relacionado con el nivel B2 de inglés!")
    
    if "historial" not in st.session_state:
        st.session_state.historial = []

    rag_chain = cargar_motor_rag()
    
    pregunta = st.text_input("Pregunta algo: ", key="input_usuario")
    
    if st.button("Enviar"):
        if pregunta:
            try:
                respuesta = rag_chain.invoke(pregunta)
                st.success("Respuesta generada con éxito.")
                st.write(respuesta)
            except Exception as e:
                st.error(f"Error al generar la respuesta: {e}")
        else :
            st.write("Por favor, ingresa una pregunta.")
            
    

if __name__ == "__main__":
    main()