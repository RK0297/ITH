import requests
import json
import logging
from typing import Dict, List, Optional
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for legal queries"""
    
    def __init__(
        self,
        vector_database,
        ollama_model: str = "qwen3:8b",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        relevance_threshold: float = 0.35 # Threshold for using RAG vs pure LLM
    ):
        self.vector_db = vector_database
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.relevance_threshold = relevance_threshold
        self.conversations = {}  # Store conversation history
        
        logger.info(f"RAG Pipeline initialized with model: {ollama_model}")
        logger.info(f"Relevance threshold: {relevance_threshold}")
    
    def check_ollama(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                return any(self.ollama_model in name for name in model_names)
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama: {e}")
            return False
    
    def retrieve_context(self, query: str, top_k: int = 5) -> Dict:
        """Retrieve relevant documents from vector database"""
        logger.info(f"Retrieving top {top_k} documents for query")
        
        results = self.vector_db.search(query, n_results=top_k)
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results.get('distances', [[]])[0]
        }
    
    def build_prompt(
        self,
        query: str,
        context_docs: List[str],
        context_metadata: List[Dict],
        conversation_history: List[Dict] = None,
        use_rag: bool = True
    ) -> str:
        """Build prompt for LLM with or without retrieved context"""
        
        if use_rag:
            # RAG mode: Use database context
            system_prompt = """You are an AI legal assistant specializing in Indian law, particularly the Indian Constitution and legal matters. 
Your role is to provide accurate, helpful, and well-cited legal information based on the provided context.

Guidelines:
1. Answer questions clearly, comprehensively, and in a conversational manner (like ChatGPT)
2. Base your answers on the provided Q&A examples from the Indian legal dataset
3. Cite the reference IDs when using specific information
4. Provide complete explanations with examples where applicable
5. Use simple language while maintaining legal accuracy
6. If asked about legal advice, remind users to consult a qualified lawyer for specific cases

Context from Indian Law Q&A Database:
"""
            
            # Add context documents with sources (now Q&A pairs)
            for i, (doc, metadata) in enumerate(zip(context_docs, context_metadata), 1):
                source_info = f"[Reference {i}]"
                
                # Extract instruction and response from metadata if available
                instruction = metadata.get('instruction', '')
                response = metadata.get('response', '')
                qa_id = metadata.get('id', str(i))
                
                if instruction and response:
                    system_prompt += f"\n{source_info} (ID: {qa_id}):\n"
                    system_prompt += f"Question: {instruction}\n"
                    system_prompt += f"Answer: {response}\n"
                else:
                    # Fallback to document content
                    system_prompt += f"\n{source_info}:\n{doc}\n"
        else:
            # Pure LLM mode: No database context, use general knowledge
            system_prompt = """You are an AI legal assistant with expertise in Indian law, the Indian Constitution, and legal matters. 

Guidelines:
1. Answer questions clearly, comprehensively, and in a conversational manner (like ChatGPT)
2. Use your general knowledge about Indian law
3. Provide complete explanations with examples where applicable
4. Use simple language while maintaining legal accuracy
5. If providing legal information, remind users to consult a qualified lawyer for specific cases
6. If you're not certain about specific legal details, acknowledge it

Note: This response is based on general legal knowledge. For specific database information, please try rephrasing your query.
"""
        
        # Add conversation history if available
        prompt = system_prompt + "\n\nConversation:\n"
        
        if conversation_history:
            for msg in conversation_history[-3:]:  # Last 3 exchanges
                role = msg['role'].capitalize()
                prompt += f"{role}: {msg['content']}\n"
        
        # Add current query
        prompt += f"User: {query}\nAssistant: "
        
        return prompt
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using Ollama"""
        try:
            url = f"{self.ollama_base_url}/api/generate"
            
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": max_tokens  # Increased for complete responses
                }
            }
            
            logger.info("Generating response from Ollama...")
            response = requests.post(url, json=payload, timeout=120)  # Increased timeout
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "I apologize, but the response generation is taking longer than expected. Please try again."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating the response. Please ensure Ollama is running with the {self.ollama_model} model."
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history for a given ID"""
        return self.conversations.get(conversation_id, [])
    
    def update_conversation_history(
        self,
        conversation_id: str,
        user_query: str,
        assistant_response: str
    ):
        """Update conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].extend([
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_response}
        ])
        
        # Keep only last 10 messages (5 exchanges)
        if len(self.conversations[conversation_id]) > 10:
            self.conversations[conversation_id] = self.conversations[conversation_id][-10:]
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        conversation_id: Optional[str] = None
    ) -> Dict:
        """
        Main query method - orchestrates the entire RAG pipeline
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            conversation_id: Optional conversation ID for context
            
        Returns:
            Dictionary with response, sources, conversation_id, and mode (rag/llm)
        """
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Step 1: Retrieve relevant context
        context = self.retrieve_context(query, top_k=top_k)
        
        # Step 2: Check relevance - use distance/similarity score
        # ChromaDB returns distances (lower = more similar)
        # If all distances are high (low similarity), use pure LLM mode
        distances = context.get('distances', [1.0] * top_k)
        avg_distance = sum(distances) / len(distances) if distances else 1.0
        
        # Convert distance to similarity score (0-1, where 1 is most similar)
        avg_similarity = 1 - avg_distance
        
        use_rag = avg_similarity >= self.relevance_threshold
        
        if use_rag:
            logger.info(f"Using RAG mode (similarity: {avg_similarity:.3f})")
        else:
            logger.info(f"Using pure LLM mode (similarity too low: {avg_similarity:.3f})")
        
        # Step 3: Get conversation history
        history = self.get_conversation_history(conversation_id)
        
        # Step 4: Build prompt (with or without RAG)
        if use_rag:
            prompt = self.build_prompt(
                query=query,
                context_docs=context['documents'],
                context_metadata=context['metadatas'],
                conversation_history=history,
                use_rag=True
            )
        else:
            prompt = self.build_prompt(
                query=query,
                context_docs=[],
                context_metadata=[],
                conversation_history=history,
                use_rag=False
            )
        
        # Step 5: Generate response with higher token limit
        response = self.generate_response(prompt, max_tokens=1500)
        
        # Step 6: Update conversation history
        self.update_conversation_history(conversation_id, query, response)
        
        # Return complete result
        return {
            'response': response,
            'documents': context['documents'] if use_rag else [],
            'metadatas': context['metadatas'] if use_rag else [],
            'conversation_id': conversation_id,
            'mode': 'rag' if use_rag else 'llm',
            'similarity_score': avg_similarity
        }

def test_rag_pipeline():
    """Test the RAG pipeline with the new Q&A format"""
    from vector_database import VectorDatabase
    
    # Initialize components
    print("Initializing Vector Database...")
    vector_db = VectorDatabase(
        persist_directory="data/vectordb",
        collection_name="legal_qa"
    )
    
    # Check if DB has data
    db_count = vector_db.collection.count()
    print(f"Vector DB contains {db_count} Q&A pairs")
    
    if db_count == 0:
        print("\n⚠️  Vector database is empty!")
        print("Please run: python vector_database.py to populate it first")
        return
    
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline(vector_database=vector_db)
    
    # Check Ollama
    if not rag.check_ollama():
        print("\n⚠️  Warning: Ollama is not running or model is not available")
        print(f"Please ensure Ollama is running and execute: ollama pull {rag.ollama_model}")
        print("\nContinuing with retrieval test only...\n")
        
        # Test retrieval only
        test_query = "What is a petition in Indian law?"
        print(f"Testing retrieval for: {test_query}")
        print("-" * 60)
        
        results = rag.retrieve_context(test_query, top_k=3)
        print(f"\nFound {len(results['documents'])} relevant Q&A pairs:\n")
        
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas']), 1):
            print(f"{i}. Q&A ID: {metadata.get('id')}")
            print(f"   Instruction: {metadata.get('instruction', '')[:100]}...")
            print(f"   Response: {metadata.get('response', '')[:150]}...")
            print()
        
        return
    
    # Test queries
    test_queries = [
        "What are the legal actions against rape",
    ]
    
    print("\n" + "="*60)
    print("Testing RAG Pipeline with Q&A Format")
    print("="*60)
    
    conversation_id = None
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-" * 60)
        
        result = rag.query(query, top_k=3, conversation_id=conversation_id)
        conversation_id = result['conversation_id']
        
        mode = result.get('mode', 'rag').upper()
        similarity = result.get('similarity_score', 0.0)
        print(f"🔍 Mode: {mode} (Similarity: {similarity:.3f})\n")
        
        print(f"💬 Response:\n{result['response']}\n")
        
        if result.get('metadatas'):
            print("📚 Sources from Q&A Database:")
            for i, metadata in enumerate(result['metadatas'], 1):
                qa_id = metadata.get('id', 'N/A')
                instruction = metadata.get('instruction', 'N/A')
                print(f"  {i}. [ID: {qa_id}] {instruction[:80]}...")
        else:
            print("📚 No database sources used (pure LLM response)")
        print()

if __name__ == "__main__":
    test_rag_pipeline()