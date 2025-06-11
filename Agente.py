# Importar bibliotecas necesarias
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# Configurar la clave de API de OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Configuración de página
st.set_page_config(
    page_title="ChatRegulación Energética",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Asistente Regulación Energía - Colombia.")
st.markdown("Carga un documento PDF y haz preguntas sobre su contenido usando IA.")

# Inicializar estado de la sesión
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

def process_pdf(file):
    """Procesar el PDF y configurar los componentes de LangChain"""
    try:
        # Leer el PDF
        pdf_reader = PdfReader(file)
        raw_text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()  # Extraer texto de cada página
            if content:
                raw_text += content

        # Verificar si se extrajo texto
        if not raw_text:
            raise ValueError("No se pudo extraer texto del PDF")

        # Dividir el texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",  # Separador de líneas
            chunk_size=800,  # Tamaño máximo de fragmentos
            chunk_overlap=200,  # Solapamiento entre fragmentos
            length_function=len,  # Función para medir longitud
        )
        texts = text_splitter.split_text(raw_text)

        # Crear embeddings y almacén vectorial
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Explicitly specify model, no proxies
        document_search = FAISS.from_texts(texts, embeddings)  # Usar FAISS para mejor rendimiento

        # Definir rol y plantilla de prompt
        role_description = (
            "Eres un asistente de IA especializado en análisis de la regulación del mercado "
            "de energía en Colombia. Proporciona respuestas precisas y detalladas basadas en el documento proporcionado."
        )
        QA_CHAIN_PROMPT = PromptTemplate.from_template(
            role_description + "\n\nPregunta: {question}\nContexto: {context}\nRespuesta:"
        )

        # Inicializar el modelo de lenguaje
        llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")

        # Configurar memoria del chat
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Crear cadena conversacional
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=document_search.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        return None

# Sidebar para configuración
with st.sidebar:
    st.header("📋 Configuración")
    
    # Componente para cargar archivo PDF
    uploaded_file = st.file_uploader(
        "Carga un archivo PDF",
        type=["pdf"],
        help="Sube un documento PDF para analizarlo con IA"
    )
    
    # Botón para limpiar chat
    if st.button("🗑️ Limpiar Chat"):
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.session_state.processed_file = None
        st.rerun()

# Procesar el PDF cuando se carga (solo si es un archivo nuevo)
if uploaded_file and (st.session_state.processed_file != uploaded_file.name):
    with st.spinner("🔄 Procesando PDF..."):
        st.session_state.qa_chain = process_pdf(uploaded_file)
        st.session_state.processed_file = uploaded_file.name
    
    if st.session_state.qa_chain:
        st.success("✅ ¡PDF procesado correctamente! Ahora puedes hacer preguntas.")
    else:
        st.error("❌ No se pudo procesar el PDF. Verifica el archivo e intenta de nuevo.")

# Área principal de chat
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Conversación")
    
    # Contenedor para el chat con scroll
    chat_container = st.container()
    
    with chat_container:
        # Mostrar historial de chat
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(f"**Tú:** {message['content']}")
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**Asistente:** {message['content']}")

with col2:
    if st.session_state.qa_chain:
        st.success("📄 PDF cargado")
        st.info(f"📊 {len(st.session_state.chat_history)} mensajes")
    else:
        st.warning("⚠️ Sin PDF")

# Entrada para nueva pregunta
if st.session_state.qa_chain:
    with st.form(key="question_form", clear_on_submit=True):
        user_input = st.text_area(
            "Haz una pregunta sobre el PDF:",
            height=100,
            placeholder="Ejemplo: ¿Cuáles son los principales puntos de la regulación CREG 071-2006?"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submit_button = st.form_submit_button("📤 Enviar", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("🗑️ Limpiar", use_container_width=True)

    # Procesar la pregunta
    if submit_button and user_input.strip():
        # Agregar pregunta al historial
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input.strip()
        })

        # Generar respuesta
        with st.spinner("🤔 Generando respuesta..."):
            try:
                response = st.session_state.qa_chain({
                    "question": user_input.strip()
                })
                
                answer = response.get("answer", "No se pudo generar una respuesta.")
                
                # Agregar respuesta al historial
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer
                })
                
                # Recargar para mostrar la nueva conversación
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error al generar respuesta: {str(e)}")
                
    elif submit_button and not user_input.strip():
        st.warning("⚠️ Por favor, ingresa una pregunta.")
        
    elif clear_button:
        st.session_state.chat_history = []
        st.rerun()
        
else:
    st.info("👆 **Primero carga un archivo PDF usando el panel lateral**")
    
    # Ejemplos de preguntas
    st.subheader("💡 Ejemplos de preguntas que puedes hacer:")
    examples = [
        "¿Cuál es el objetivo principal de esta regulación?",
        "¿Qué entidades están involucradas en este proceso?",
        "¿Cuáles son las obligaciones principales establecidas?",
        "¿Qué sanciones se contemplan?",
        "¿Cuándo entra en vigencia esta regulación?"
    ]
    
    for example in examples:
        st.markdown(f"• {example}")

# Footer
st.markdown("---")
