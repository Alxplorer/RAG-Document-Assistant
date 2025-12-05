import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

DATA_PATH = "documentos/" #para definir la ruta del documento
DB_PATH = "db/chroma/" #para definir la ruta de la base de datos

load_dotenv()  # Cargar variables de entorno desde el archivo .env

def cargar_documentos(path_al_archivo):
    documentos = []
    for filename in os.listdir(path_al_archivo):
        if filename.endswith('.pdf'):
            filepath = os.path.join(path_al_archivo, filename)
            loader = PyPDFLoader(filepath)
            documentos.extend(loader.load())
    return documentos

def main():
    documentos = cargar_documentos(DATA_PATH)

    text_splitter = RecursiveCharacterTextSplitter(    #para definir el divisor de texto
        chunk_size=1000, 
        chunk_overlap=100,  
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documentos) #para dividir los documentos en fragmentos

    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small") #para definir el modelo de embeddings

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"Ingesta completada... {len(chunks)} chunks guardados en {DB_PATH}")
    return vectordb

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 
    main()