""" 
Production-ready RAG pipeline for document/information based question-answering using OpenAI embeddings and FAISS.
Uses multiple sources, caching and observability.
"""
import os
from pydoc import doc
from typing import List, Dict, Any, Optional
import logging
import time
from datetime import datetime

# Langchain imports
from app.observability import TOKENS_USED, VECTOR_STORE_SIZE
from langchain_openai import OpenAIEmbeddings  
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

# Local imports
from app.config import CONFIG
from app.information_loader import InformationLoader

logger = logging.getLogger(__name__)

class ProductionRAGPipeline:
    """
    Production grade RAG pipeline for document/information based question-answering 
    - with mulitiple sources
    - better text splitting
    - caching mechanism
    - token tracking
    - error handling
    """

    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.information_loader = InformationLoader()
        self.vector_store = None
        self.qa_chain = None

        # Initialize vector store and QA chain embeddings and FAISS
        self.embeddings = OpenAIEmbeddings(
            model=CONFIG['embeddings']['model'], 
        )
          
    def create_vector_store(self, sources_list: List[str]):
        """
        Creates a vector store from multiple sources (PDFs, web pages, local files.

        Args:
            sources_list (List[str]): List of source paths or URLs to load information from.
        """
        import inspect
        import traceback
        logger.info(f"Creating vector store from {len(sources_list)} sources.")

        all_texts = []
        all_metadatas = []

        for i, source in enumerate(sources_list):
            try:
                # Load and extract text from the source
                content = self.information_loader.load_information(source)

                # Split the text into chunks for information retrieval
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CONFIG['rag']['chunk_size'],
                    chunk_overlap=CONFIG['rag']['chunk_overlap'],
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )

                texts = text_splitter.split_text(content)

                # Ensure all chunks are strings (in case of any non-string content)
                for j, chunk in enumerate(texts):
                    if not isinstance(chunk, str):
                        logger.warning(f"Chunk {j} is non-string of type {type(chunk)}. Converting...")
                        texts[j] = str(chunk)
 
                # Add metadata to each chunk for better traceability
                for chunk in texts:
                    all_texts.append(chunk)
                    all_metadatas.append({
                        "source": source,
                        "chunk_index": len(all_texts) - 1,
                        "source_type": self.get_source_type(source)
                    })

                logger.info(f"Processed {source} with {len(texts)} chunks.")
            
            except Exception as e:
                logger.error(f"Failed to process source {source}: {str(e)}")
                continue
        
        if not all_texts:
            raise ValueError("No valid text loaded from sources to create vector store.")
        
        # Update vector store size metric
        VECTOR_STORE_SIZE.set(len(all_texts))
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            texts=all_texts, 
            embedding=self.embeddings,
            metadatas=all_metadatas
        )

        # Create QA chain
        self.qa_chain = self._create_qa_chain()

        logger.info(f"Vector store created successfully with {len(all_texts)} chunks.")

    def get_source_type(self, source: str) -> str:
        """ 
        Helper method to determine source type based on file extension or URL pattern.
        """
        if source.startswith('s3://'):
            return 's3'
        elif source.startswith(('http://', 'https://')):
            return 'web'
        else:
            return 'local'
        
    def _create_qa_chain(self):
        """
        Create question-answering (qa) chain with a better prompt template and error handling.
        """    
        prompt_template = """You are a helpful assistant that answers questions based on the provided information. Use the following context to answer the question. 
        
        Context:
        {context}

        Question: {question}

        Answer in a concise and accurate manner, and only use the provided context to answer. If the answer is not in the context, say "I don't know".
        """
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

        llm = OpenAI(
            openai_api_key=self.openai_api_key,
            model=CONFIG['llm']['model'],
            temperature=CONFIG['llm']['temperature']
        )

        self.qa_chain = load_qa_chain(
            llm, 
            chain_type="stuff", 
            prompt=PROMPT
        )
        
    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Query the rag pipeline with a question and get an answer along with sources if requested. 
        Observability and error handling included.
        
        Args:
            question (str): The question to ask the RAG pipeline.
            return_sources (bool): Whether to return the sources used for answering the question.

        Returns:
            Dict[str, Any]: A dictionary containing the answer and optionally the sources and metrics.

        """
        start_time = time.time()

        if not self.vector_store or not self.qa_chain:
            raise ValueError("Vector store or QA chain not initialized. Create the vector store first.")
        
        try:
            # Retrieve relevant documents from the vector store
            docs = self.vector_store.similarity_search(question, k=CONFIG['rag']['top_k'])

            # Track tokens usage with OpenAI callback
            with get_openai_callback() as cb:
                answer = self.qa_chain.run(input_documents=docs, question=question)

                # Increment token usage metric
                TOKENS_USED.inc(cb.total_tokens)
                # Prepare response with answer and sources if requested
                response = {
                    "answer": answer,
                    "processing_time": time.time() - start_time,
                    "tokens_used": cb.total_tokens,
                    "cost_estimate": cb.total_cost
                }

                if return_sources:
                    sources = []
                    for i, doc in enumerate(docs):
                        source_info = {
                            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content, 
                            "source": doc.metadata.get("source", "Unknown"),
                            "source_type": doc.metadata.get("source_type", "Unknown")
                        }

                        sources.append(source_info)
                    response["sources"] = sources

                logger.info(f"Question answered in {response['processing_time']:.2f} seconds with {response['tokens_used']} tokens used.")
                return response
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise

# ==================== LEGACY FUNCTIONS (for backward compatibility) ====================
    def create_vector_store_legacy(pdf_path: str, openai_api_key: str) -> FAISS:
        """
        Legacy function for creating vector store from a single PDF file. This can be used for backward compatibility or testing purposes.

        Args:
            pdf_path (str): Path to the PDF file to load and create vector store from.
            openai_api_key (str): OpenAI API key for creating embeddings.

        Returns:
            faiss.FAISS: The created FAISS vector store with the PDF content.
        """
        pipeline = ProductionRAGPipeline(openai_api_key)
        pipeline.create_vector_store([pdf_path])
        return pipeline.vector_store
        
    def get_qa_chain_legacy(openai_api_key: str):
        """
        Legacy function for creating QA chain. This can be used for backward compatibility or testing purposes.
        """
        pipeline = ProductionRAGPipeline(openai_api_key)
        pipeline._create_qa_chain()
        return pipeline.qa_chain
    
    def query_vector_store_legacy(vector_store: FAISS, chain: Any, question: str) -> str:
        """
        Legacy function for querying the vector store and getting an answer. This can be used for backward compatibility or testing purposes.
        """
        docs = vector_store.similarity_search(question)
        return chain.run(input_documents=docs, question=question)