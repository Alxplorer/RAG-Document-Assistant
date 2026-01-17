
import streamlit as st 

from langchain_chroma import Chroma 

from langchain_openai import OpenAIEmbeddings 

from langchain_openai import ChatOpenAI 

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

from dotenv import load_dotenv
load_dotenv()  # Cargar variables de entorno desde el archivo .env

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}, score_threshold=0.6) #para definir el recuperador de documentos
    llm = ChatOpenAI(temperature=0.1, model_name=LLM_MODEL, presence_penalty=0) #para definir el modelo de lenguaje
    parser = StrOutputParser()  #para convertir la salida en un string
    
    prompt = ChatPromptTemplate.from_template(
        """
        Eres un Tutor de Inglés experto y empático. Ayudas a estudiantes de nivel B1 a avanzar hacia un nivel B2.
        Tu objetivo es explicar gramática y vocabulario de forma clara,  corregir errores y dar ejemplos prácticos.
        Usa un tono paciente y que motive. 
        Contexto de las guías de estudio:{context}
        Pregunta = {question}
         Instrucciones: 
         1. Responde únicamente usando el contexto proporcionado anteriormente para responder la pregunta.  
         2. Evalúa si la pregunta es relevante para el contexto dado.
         3. Si NO es relevante, responde: "No tengo información sobre ese tema en mis materiales de aprendizaje."
         4. Si es relevante, responde usando SOLO la información proporcionada.
         5. Usa un inglés claro, apropiado para estudiantes de nivel B1-B2.
         6. Incluye explicaciones de vocabulario si es necesario.
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
    st.markdown("Creado con arquitectura RAG usando LangChain, Chroma y OpenAI...")
    st.subheader("🦜🔗 Chatbot Tutor de :red[Inglés Nivel B2]")
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