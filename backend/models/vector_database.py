import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Manage ChromaDB vector database for legal Q&A documents"""
    
    def __init__(
        self, 
        persist_directory: str = "data/vectordb",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "legal_qa"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        
        logger.info(f"Vector database initialized at {self.persist_directory}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            count = collection.count()
            logger.info(f"Loaded existing collection '{self.collection_name}' with {count} documents")
            return collection
        except Exception:
            logger.info(f"Creating new collection '{self.collection_name}'")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Indian Law Q&A Dataset - Instruction-Response pairs"}
            )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def add_documents(self, qa_pairs: List[Dict], batch_size: int = 100):
        """Add Q&A pairs to vector database
        
        Args:
            qa_pairs: List of dicts with 'id', 'Instruction', 'Response' keys
            batch_size: Number of documents to process in each batch
        """
        logger.info(f"Adding {len(qa_pairs)} Q&A pairs to vector database...")
        
        # Prepare data
        ids = []
        documents = []
        metadatas = []
        
        for qa in qa_pairs:
            # Create unique ID
            doc_id = f"qa_{qa['id']}"
            ids.append(doc_id)
            
            # Combine Instruction and Response for embedding
            # This helps the model understand context better
            combined_text = f"Question: {qa['Instruction']}\n\nAnswer: {qa['Response']}"
            documents.append(combined_text)
            
            # Store metadata separately for retrieval
            metadata = {
                'id': str(qa['id']),
                'instruction': qa['Instruction'][:500],  # Truncate for storage
                'response': qa['Response'][:500],  # Truncate for storage
                'instruction_length': str(len(qa['Instruction'])),
                'response_length': str(len(qa['Response']))
            }
            metadatas.append(metadata)
        
        # Add to database in batches
        logger.info(f"Processing in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding batches"):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_metadata = metadatas[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = self.generate_embeddings(batch_docs)
            
            # Add to collection
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metadata,
                embeddings=batch_embeddings
            )
        
        logger.info(f"✅ Successfully added {len(documents)} Q&A pairs")
        logger.info(f"Total documents in collection: {self.collection.count()}")
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_dict: Dict = None
    ) -> Dict:
        """Search for similar Q&A pairs based on query
        
        Args:
            query: The search query (can be a question or keywords)
            n_results: Number of top results to return
            filter_dict: Optional metadata filters
            
        Returns:
            Dictionary with search results including documents and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search parameters
        search_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results
        }
        
        # Add filter if provided
        if filter_dict:
            search_params["where"] = filter_dict
        
        # Perform search
        results = self.collection.query(**search_params)
        
        return results
    
    def search_by_instruction(self, instruction: str, n_results: int = 5) -> List[Dict]:
        """Search specifically by instruction/question similarity
        
        Returns formatted list of Q&A pairs
        """
        results = self.search(instruction, n_results=n_results)
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0],
                results['distances'][0] if 'distances' in results else [0] * len(results['documents'][0])
            )):
                formatted_results.append({
                    'rank': i + 1,
                    'id': metadata.get('id'),
                    'instruction': metadata.get('instruction'),
                    'response': metadata.get('response'),
                    'full_text': doc,
                    'similarity_score': 1 - distance  # Convert distance to similarity
                })
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        count = self.collection.count()
        
        # Get sample documents to analyze
        sample_size = min(100, count)
        sample = self.collection.get(limit=sample_size)
        
        stats = {
            "total_qa_pairs": count,
            "collection_name": self.collection_name,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        # Calculate average lengths if we have samples
        if sample['metadatas']:
            instruction_lengths = []
            response_lengths = []
            
            for metadata in sample['metadatas']:
                if 'instruction_length' in metadata:
                    instruction_lengths.append(int(metadata['instruction_length']))
                if 'response_length' in metadata:
                    response_lengths.append(int(metadata['response_length']))
            
            if instruction_lengths:
                stats['avg_instruction_length'] = sum(instruction_lengths) // len(instruction_lengths)
            if response_lengths:
                stats['avg_response_length'] = sum(response_lengths) // len(response_lengths)
        
        return stats
    
    def reset_database(self):
        """Reset the entire database (use with caution!)"""
        logger.warning("Resetting database...")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info("Database reset complete")

def load_qa_data(filename: str = "legal_data_all.json", data_dir: str = "../scrapers/data/raw") -> List[Dict]:
    """Load Q&A data from JSON file
    
    Args:
        filename: Name of the JSON file
        data_dir: Directory containing the data file
        
    Returns:
        List of Q&A pairs with id, Instruction, Response keys
    """
    filepath = Path(data_dir) / filename
    
    logger.info(f"Loading Q&A data from: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✅ Loaded {len(data)} Q&A pairs")
        
        # Validate format
        if data and isinstance(data, list):
            first_item = data[0]
            required_keys = {'id', 'Instruction', 'Response'}
            if not required_keys.issubset(first_item.keys()):
                logger.warning(f"Data format unexpected. Expected keys: {required_keys}, Got: {first_item.keys()}")
        
        return data
        
    except FileNotFoundError:
        logger.error(f"❌ File not found: {filepath}")
        logger.info("Please run the data loader first: python ../scrapers/dt.py")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error decoding JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return []

def main():
    """Main function to setup vector database with Q&A pairs"""
    print("=" * 60)
    print("Setting up Vector Database for Indian Law Q&A")
    print("=" * 60)
    
    # Initialize database
    db = VectorDatabase(
        persist_directory="data/vectordb",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="legal_qa"
    )
    
    # Check if database is empty
    current_count = db.collection.count()
    
    if current_count == 0:
        print("\n📂 Database is empty. Loading Q&A data...")
        
        # Load Q&A data
        qa_data = load_qa_data("legal_data_all.json", "../scrapers/data/raw")
        
        if not qa_data:
            print("❌ No Q&A data found. Please run the data loader first:")
            print("   python ../scrapers/dt.py")
            return
        
        print(f"📊 Found {len(qa_data)} Q&A pairs to add")
        
        # Ask user if they want to add all or a subset
        choice = input(f"\nAdd all {len(qa_data)} Q&A pairs? (y/n): ").lower().strip()
        
        if choice != 'y':
            try:
                num = int(input("Enter number of Q&A pairs to add: "))
                qa_data = qa_data[:num]
            except:
                print("Invalid input. Adding first 1000 pairs...")
                qa_data = qa_data[:1000]
        
        # Add to database
        print(f"\n🔄 Adding {len(qa_data)} Q&A pairs to vector database...")
        db.add_documents(qa_data, batch_size=100)
        
    else:
        print(f"\n✅ Database already contains {current_count} Q&A pairs")
        
        choice = input("\nAdd more data? (y/n): ").lower().strip()
        if choice == 'y':
            qa_data = load_qa_data("legal_data_all.json", "../scrapers/data/raw")
            if qa_data:
                # Skip already added
                start_id = current_count
                new_data = [qa for qa in qa_data if qa['id'] >= start_id]
                
                if new_data:
                    print(f"📊 Found {len(new_data)} new Q&A pairs")
                    db.add_documents(new_data, batch_size=100)
                else:
                    print("No new data to add.")
    
    # Display stats
    print("\n" + "=" * 60)
    print("📊 Database Statistics")
    print("=" * 60)
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search
    print("\n" + "=" * 60)
    print("🔍 Testing Search Functionality")
    print("=" * 60)
    
    test_queries = [
        "What is a petition in Indian law?",
        "How to file a PIL?",
        "Article 21 rights"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        results = db.search_by_instruction(query, n_results=2)
        
        for result in results:
            print(f"\n  Rank {result['rank']} (Similarity: {result['similarity_score']:.3f})")
            print(f"  Q: {result['instruction'][:150]}...")
            print(f"  A: {result['response'][:150]}...")
    
    print("\n" + "=" * 60)
    print("✅ Vector Database Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Use the search functions in your RAG pipeline")
    print("2. Run: python rag_pipeline.py (if available)")
    print("3. Test with different queries")

if __name__ == "__main__":
    main()