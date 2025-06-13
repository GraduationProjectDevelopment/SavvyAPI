import os
import logging
from dotenv import load_dotenv
from langchain_fireworks import Fireworks, FireworksEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from supabase import create_client
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv("API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not all([api_key, supabase_url, supabase_key]):
    logger.error(
        "One or more required environment variables are missing: API_KEY, SUPABASE_URL, SUPABASE_KEY"
    )

# Initialize LLM and embeddings
llm = Fireworks(
    api_key=api_key,
    model="accounts/fireworks/models/deepseek-v3",
    temperature=1.0,
    max_tokens=512,     
    top_p=0.9,
    frequency_penalty=0.5
)
embeddings = FireworksEmbeddings(api_key=api_key)

# OS-agnostic PDF file paths
pdf_files = [
    os.path.join("app", "Pdfs", "1_55_ways_to_save.pdf"),
    os.path.join("app", "Pdfs", "40MoneyManagementTips.pdf"),
    os.path.join("app", "Pdfs", "beginners-guide-to-saving-2024.pdf"),
    os.path.join("app", "Pdfs", "How-to-Manage-your-Finances.pdf"),
    os.path.join("app", "Pdfs", "pdf_50_20_30.pdf"),
    os.path.join("app", "Pdfs", "Personal-Finance-Management-Handbook.pdf"),
    os.path.join("app", "Pdfs", "reach-my-financial-goals.pdf"),
    os.path.join("app", "Pdfs", "tips-to-manage-your-money.pdf"),
]


class FinancialChatbot:
    def __init__(self):
        self.pdf_vector_store = None
        self.user_sessions = {}
        self._initialize_pdf_knowledge_base()

    def _initialize_pdf_knowledge_base(self):
        try:
            logger.info("Loading PDF documents...")
            documents = []
            for pdf in pdf_files:
                if os.path.exists(pdf):
                    logger.info(f"Loading PDF: {pdf}")
                    loader = PyPDFLoader(pdf)
                    documents.extend(loader.load())
                else:
                    logger.warning(f"PDF file not found: {pdf}")

            if not documents:
                logger.error("No PDF documents loaded!")
                return

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")

            # Embed all chunks at once (or batch if needed)
            texts = [chunk.page_content for chunk in chunks]
            pdf_embeddings = embeddings.embed_documents(texts)

            if pdf_embeddings and len(pdf_embeddings) == len(texts):
                # Use from_texts which is common FAISS API method
                self.pdf_vector_store = FAISS.from_texts(
                    texts=texts, embedding=embeddings
                )
                logger.info("PDF knowledge base initialized successfully")
            else:
                logger.error("Failed to create PDF embeddings properly")
        except Exception as e:
            logger.error(f"Error initializing PDF knowledge base: {e}")

    def _fetch_user_transactions(self, user_id: str):
        """Fetch transactions directly via Supabase client."""
        supabase = create_client(supabase_url, supabase_key)
        try:
            response = (
                supabase.table("transactions")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )
            logger.info(
                f"Fetched {len(response.data) if response.data else 0} transactions for user {user_id}"
            )
            return response.data
        except Exception as e:
            logger.error(f"Supabase query failed for user {user_id}: {e}")
            raise Exception(f"Failed to fetch transactions: {str(e)}")

    def _create_user_transaction_vector_store(self, user_id: str):
        try:
            data = self._fetch_user_transactions(user_id)
            if not data:
                logger.info(f"No transactions found for user {user_id}")
                return None

            raw_transaction_docs = [
                Document(
                    page_content=(
                        f"Transaction ID: {tx.get('transaction_id', 'Unknown')}\n"
                        f"Date: {tx.get('created_at', 'Unknown date')}\n"
                        f"Type: {tx.get('transaction_type', 'Unknown')}\n"
                        f"Amount: {tx.get('amount', 0)} EGP\n"
                        f"Category ID: {tx.get('category_id', 'Unknown')}\n"
                        f"Description: {tx.get('description', 'No description')}"
                    ),
                    metadata={
                        "source": "transactions",
                        "transaction_id": tx.get("transaction_id"),
                        "user_id": user_id,
                    },
                )
                for tx in data
            ]

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunked_docs = []
            for doc in raw_transaction_docs:
                split_chunks = splitter.split_documents([doc])
                chunked_docs.extend(split_chunks)

            if not chunked_docs:
                logger.error(f"No chunks generated for user {user_id}")
                return None

            transaction_texts = [chunk.page_content for chunk in chunked_docs]

            transaction_embeddings = embeddings.embed_documents(transaction_texts)
            if not transaction_embeddings or len(transaction_embeddings) != len(
                transaction_texts
            ):
                logger.error(
                    f"Embedding failed or incomplete for transactions of user {user_id}"
                )
                return None

            transaction_vector_store = FAISS.from_texts(
                texts=transaction_texts, embedding=embeddings
            )
            logger.info(
                f"Created transaction vector store for user {user_id} with "
                f"{len(chunked_docs)} chunks from {len(raw_transaction_docs)} transactions"
            )
            return transaction_vector_store

        except Exception as e:
            logger.error(
                f"Error creating transaction vector store for user {user_id}: {e}"
            )
            return None

    def _get_or_create_user_session(self, user_id: str):
        if user_id not in self.user_sessions:
            logger.info(f"Creating new session for user {user_id}")

            transaction_vector_store = self._create_user_transaction_vector_store(
                user_id
            )

            retrievers = []
            weights = []

            # Always include PDF retriever if available
            if self.pdf_vector_store:
                pdf_retriever = self.pdf_vector_store.as_retriever(
                    search_kwargs={"k": 5}
                )
                retrievers.append(pdf_retriever)
                weights.append(0.5)  # 50% weight to general financial advice

            # Include transaction retriever if user has transactions
            if transaction_vector_store:
                transaction_retriever = transaction_vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
                retrievers.append(transaction_retriever)
                weights.append(0.5)  # 50% weight to user transactions

            if len(retrievers) > 1:
                combined_retriever = EnsembleRetriever(
                    retrievers=retrievers, weights=weights
                )
            elif len(retrievers) == 1:
                combined_retriever = retrievers[0]
            else:
                logger.error(f"No retrievers available for user {user_id}")
                return None

            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )

            prompt_template = PromptTemplate.from_template(
                """
You are a helpful financial assistant specifically helping this user with their personal finances.
Use the following context to answer the question in a professional and practical way.

Context includes:
1. General financial advice from professional resources
2. The user's personal transaction history (if available)

When answering:
- If the question is about the user's spending patterns, reference their specific transaction data
- If it's about general financial advice, use the professional resources
- Always personalize your response to the user's situation when possible
- Be encouraging and supportive
- Provide actionable advice

Context:
{context}

Chat History:
{chat_history}

Current Question:
{question}

Provide a helpful, personalized response:
"""
            )

            conversational_rag = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=combined_retriever,
                chain_type="stuff",
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                output_key="answer",
            )

            self.user_sessions[user_id] = {
                "chain": conversational_rag,
                "memory": memory,
                "transaction_store": transaction_vector_store,
                "created_at": None,  # Could add timestamp if needed
            }

            logger.info(f"Session created successfully for user {user_id}")

        return self.user_sessions[user_id]

    def get_chat_response(self, question: str, user_id: str) -> Dict[str, any]:
        if not user_id:
            return {"error": "User ID is required", "answer": None}

        try:
            user_session = self._get_or_create_user_session(user_id)
            if not user_session:
                return {"error": "Failed to create user session", "answer": None}

            result = user_session["chain"].invoke({"question": question})
            logger.info(f"Answer for user {user_id}: {result.get('answer', '')}")

            return {
                "answer": result["answer"],
            }

        except Exception as e:
            logger.error(f"Error getting chat response for user {user_id}: {e}")
            return {"error": str(e), "answer": None}

    def refresh_user_data(self, user_id: str):
        if user_id in self.user_sessions:
            logger.info(f"Refreshing data for user {user_id}")
            del self.user_sessions[user_id]


# Initialize the chatbot instance
financial_chatbot = FinancialChatbot()


# Main API function
def get_chat_response(question: str, user_id: str) -> Dict[str, any]:
    logger.info(f"API call - question: {question}, user_id: {user_id}")
    response = financial_chatbot.get_chat_response(question, user_id)
    logger.info(f"API response: {response}")
    return response
