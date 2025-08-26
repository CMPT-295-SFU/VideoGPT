"""
Script to read vectors and metadata from Pinecone and create a local index.
This script provides functionality to:
1. Read all vectors from a Pinecone index
2. Create a local FAISS index for fast similarity search
3. Save/load the local index with metadata
4. Perform similarity searches on the local index
"""

import os
import json
import pickle
import math
from typing import List, Dict, Any, Tuple, Optional
from pinecone import Pinecone
from tqdm import tqdm
import fire


class PineconeToLocal:
    def __init__(self, pinecone_api_key: str = None, host: str = None):
        """
        Initialize the PineconeToLocal converter.
        
        Args:
            pinecone_api_key: Pinecone API key (if None, will use PINECONE_API_KEY env var)
            host: Pinecone host URL (optional)
        """
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.host = host or os.getenv("PINECONE_HOST")
        self.local_index = None
        self.metadata_store = {}
        self.vector_ids = []
        self.vectors = []  # Store vectors as Python lists
        self.vector_dim = 0
        
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable or pass it as parameter.")
    
    def connect_to_pinecone(self, index_name: str):
        """Connect to Pinecone index using new API."""
        try:
            # Initialize Pinecone client
            pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Get the index - use host if provided
            if self.host:
                self.pinecone_index = pc.Index(index_name, host=self.host)
                print(f"Successfully connected to Pinecone index: {index_name} at host: {self.host}")
            else:
                self.pinecone_index = pc.Index(index_name)
                print(f"Successfully connected to Pinecone index: {index_name}")
            
            # Get index stats
            stats = self.pinecone_index.describe_index_stats()
            print(f"Index stats: {stats}")
            return True
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            return False
    
    def fetch_all_vectors(self, namespace: str = "", batch_size: int = 100, top_k: int = 10000):
        """
        Fetch all vectors from Pinecone index using systematic approach.
        
        Args:
            namespace: Pinecone namespace to fetch from
            batch_size: Number of vectors to fetch in each batch
            top_k: Maximum number of vectors to fetch
        
        Returns:
            Tuple of (vectors, metadata, ids)
        """
        print(f"Fetching vectors from namespace: '{namespace}'")
        
        all_vectors = []
        all_metadata = []
        all_ids = []
        seen_ids = set()  # Track IDs to avoid duplicates
        
        try:
            # First, check namespace stats to see how many vectors are actually there
            stats = self.pinecone_index.describe_index_stats()
            expected_count = 0
            
            if namespace and hasattr(stats, 'namespaces') and stats.namespaces:
                if namespace in stats.namespaces:
                    expected_count = stats.namespaces[namespace].vector_count
                    print(f"Expected vectors in namespace '{namespace}': {expected_count}")
                else:
                    print(f"Namespace '{namespace}' not found. Available: {list(stats.namespaces.keys())}")
                    return [], [], []
            else:
                expected_count = stats.total_vector_count
                print(f"Total vectors in index: {expected_count}")
            
            # Don't try to fetch more than available
            target_count = min(top_k, expected_count)
            print(f"Target to fetch: {target_count} vectors")
            
            # Use systematic approach with varied query vectors
            import random
            
            # Try multiple strategies to get comprehensive coverage
            strategies = [
                # Strategy 1: Use zero vector
                lambda: [0.0] * 1536,
                # Strategy 2: Use small random values
                lambda: [random.uniform(-0.01, 0.01) for _ in range(1536)],
                # Strategy 3: Use larger random values
                lambda: [random.uniform(-0.1, 0.1) for _ in range(1536)],
                # Strategy 4: Use normalized random vector
                lambda: self._normalize_vector([random.gauss(0, 1) for _ in range(1536)]),
                # Strategy 5: Use sparse vector (mostly zeros with few non-zero)
                lambda: self._sparse_vector(1536, 10),
            ]
            
            strategy_idx = 0
            consecutive_no_new = 0
            max_consecutive_no_new = 3
            
            while len(all_vectors) < target_count and strategy_idx < len(strategies) * 3:
                # Select strategy
                current_strategy = strategies[strategy_idx % len(strategies)]
                query_vector = current_strategy()
                
                try:
                    query_result = self.pinecone_index.query(
                        vector=query_vector,
                        top_k=min(batch_size, target_count),  # Always request full batch size
                        namespace=namespace,
                        include_values=True,
                        include_metadata=True
                    )
                    
                    if not query_result.matches:
                        print(f"No matches found with strategy {strategy_idx % len(strategies)}")
                        strategy_idx += 1
                        continue
                    
                    new_vectors_count = 0
                    for match in query_result.matches:
                        if match.id not in seen_ids:
                            all_vectors.append(match.values)
                            all_metadata.append(match.metadata or {})
                            all_ids.append(match.id)
                            seen_ids.add(match.id)
                            new_vectors_count += 1
                    
                    if new_vectors_count == 0:
                        consecutive_no_new += 1
                        print(f"Strategy {strategy_idx % len(strategies)}: No new vectors (consecutive: {consecutive_no_new})")
                        
                        if consecutive_no_new >= max_consecutive_no_new:
                            strategy_idx += 1
                            consecutive_no_new = 0
                    else:
                        consecutive_no_new = 0
                        print(f"Strategy {strategy_idx % len(strategies)}: Found {new_vectors_count} new vectors (total: {len(all_vectors)}/{target_count})")
                
                except Exception as e:
                    print(f"Error with strategy {strategy_idx % len(strategies)}: {e}")
                    strategy_idx += 1
                    continue
            
            print(f"Total unique vectors fetched: {len(all_vectors)} out of expected {expected_count}")
            
            if len(all_vectors) < expected_count * 0.8:  # If we got less than 80% of expected
                print(f"⚠️  Warning: Only fetched {len(all_vectors)} out of {expected_count} expected vectors")
                print("   This might be due to Pinecone's query limitations or clustering")
            
            return all_vectors, all_metadata, all_ids
            
        except Exception as e:
            print(f"Error fetching vectors: {e}")
            return [], [], []
    
    def _normalize_vector(self, vector):
        """Normalize a vector to unit length."""
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            return [x / magnitude for x in vector]
        return vector
    
    def _sparse_vector(self, dim, num_nonzero):
        """Create a sparse vector with only a few non-zero elements."""
        import random
        vector = [0.0] * dim
        indices = random.sample(range(dim), min(num_nonzero, dim))
        for idx in indices:
            vector[idx] = random.uniform(-1, 1)
        return vector
    
    def _fetch_vectors_alternative(self, namespace: str, top_k: int):
        """Alternative method to fetch vectors - this method is now integrated into main fetch."""
        # This method is kept for compatibility but the main method now handles everything
        print("Using main fetch method...")
        return [], [], []
    
    def create_local_index(self, vectors: List[List[float]], metadata: List[Dict], ids: List[str]):
        """
        Create a local index from vectors using pure Python (no numpy/multiarray).
        
        Args:
            vectors: List of vector embeddings
            metadata: List of metadata dictionaries
            ids: List of vector IDs
        """
        if not vectors:
            print("No vectors to create index from.")
            return False
        
        try:
            # Store vectors as simple Python lists (no numpy)
            self.vectors = vectors
            self.vector_dim = len(vectors[0]) if vectors else 0
            
            print(f"Creating simple local index with {len(vectors)} vectors of dimension {self.vector_dim}")
            
            # Store metadata and IDs
            self.metadata_store = {i: metadata[i] for i in range(len(metadata))}
            self.vector_ids = ids
            
            # Create a simple index flag
            self.local_index = True
            
            print(f"Simple local index created successfully with {len(vectors)} vectors")
            return True
            
        except Exception as e:
            print(f"Error creating local index: {e}")
            return False
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors without numpy."""
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def search_local_index(self, query_vector: List[float], top_k: int = 10) -> List[Dict]:
        """
        Search the local index for similar vectors using pure Python.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of results with scores, metadata, and IDs
        """
        if not self.local_index or not hasattr(self, 'vectors'):
            print("Local index not created yet.")
            return []
        
        try:
            # Calculate similarity scores for all vectors
            similarities = []
            for i, stored_vector in enumerate(self.vectors):
                similarity = self.cosine_similarity(query_vector, stored_vector)
                similarities.append((similarity, i))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Take top_k results
            results = []
            for rank, (score, idx) in enumerate(similarities[:top_k]):
                result = {
                    'score': score,
                    'id': self.vector_ids[idx],
                    'metadata': self.metadata_store.get(idx, {}),
                    'rank': rank + 1
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error searching local index: {e}")
            return []
    
    def save_local_index(self, filepath: str):
        """Save the local index and metadata to disk without FAISS."""
        try:
            # Save all data as pickle
            index_data = {
                'vectors': getattr(self, 'vectors', []),
                'metadata_store': self.metadata_store,
                'vector_ids': self.vector_ids,
                'vector_dim': getattr(self, 'vector_dim', 0)
            }
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(index_data, f)
            
            print(f"Local index saved to {filepath}.pkl")
            return True
            
        except Exception as e:
            print(f"Error saving local index: {e}")
            return False
    
    def load_local_index(self, filepath: str):
        """Load the local index and metadata from disk."""
        try:
            # Try loading from .pkl file first
            if os.path.exists(f"{filepath}.pkl"):
                with open(f"{filepath}.pkl", 'rb') as f:
                    index_data = pickle.load(f)
                
                self.vectors = index_data.get('vectors', [])
                self.metadata_store = index_data.get('metadata_store', {})
                self.vector_ids = index_data.get('vector_ids', [])
                self.vector_dim = index_data.get('vector_dim', 0)
                self.local_index = True if self.vectors else None
                
                print(f"Local index loaded from {filepath}.pkl")
                print(f"Index contains {len(self.vectors)} vectors")
                return True
            
            else:
                print(f"No index file found at {filepath}")
                return False
            
        except Exception as e:
            print(f"Error loading local index: {e}")
            return False
    
    def get_index_stats(self):
        """Get statistics about the local index."""
        if not self.local_index:
            return "No local index created."
        
        stats = {
            'total_vectors': len(getattr(self, 'vectors', [])),
            'vector_dimension': getattr(self, 'vector_dim', 0),
            'index_type': 'SimpleIndex (no numpy)',
            'metadata_entries': len(self.metadata_store),
            'sample_metadata': list(self.metadata_store.values())[:3] if self.metadata_store else []
        }
        
        return stats


def download_and_create_local_index(
    index_name: str = "295-vstore",
    namespace: str = "Slides",
    output_path: str = "local_index",
    top_k: int = 10000,
    host: str = "https://295-vstore-51a00a2.svc.aped-4627-b74a.pinecone.io"
):
    """
    Download vectors from Pinecone and create a local index.
    
    Args:
        index_name: Name of the Pinecone index
        namespace: Namespace to download from
        output_path: Path to save the local index
        top_k: Maximum number of vectors to download
        host: Pinecone host URL
    """
    converter = PineconeToLocal(host=host)
    
    # Connect to Pinecone
    if not converter.connect_to_pinecone(index_name):
        return False
    
    # Fetch vectors
    print("Fetching vectors from Pinecone...")
    vectors, metadata, ids = converter.fetch_all_vectors(namespace=namespace, top_k=top_k)
    
    if not vectors:
        print("No vectors were fetched.")
        return False
    
    # Create local index
    print("Creating local index...")
    if not converter.create_local_index(vectors, metadata, ids):
        return False
    
    # Save local index
    print("Saving local index...")
    if not converter.save_local_index(output_path):
        return False
    
    # Display stats
    stats = converter.get_index_stats()
    print(f"\nIndex Statistics:")
    print(f"Total vectors: {stats['total_vectors']}")
    print(f"Vector dimension: {stats['vector_dimension']}")
    print(f"Metadata entries: {stats['metadata_entries']}")
    
    if stats['sample_metadata']:
        print(f"\nSample metadata:")
        for i, sample in enumerate(stats['sample_metadata']):
            print(f"  {i+1}. {sample}")
    
    return True


def search_local_index(
    query_text: str,
    index_path: str = "local_index",
    top_k: int = 5
):
    """
    Search the local index with a text query.
    
    Args:
        query_text: Text to search for
        index_path: Path to the local index
        top_k: Number of results to return
    """
    # Load the local index
    converter = PineconeToLocal()
    if not converter.load_local_index(index_path):
        return
    
    # For this demo, we'll need to embed the query text
    # You would typically use the same embedding model used to create the vectors
    print(f"To search with text query '{query_text}', you need to:")
    print("1. Use the same embedding model (OpenAI text-embedding-ada-002)")
    print("2. Convert the text to a vector")
    print("3. Then call converter.search_local_index(vector, top_k)")
    
    # Show index stats
    stats = converter.get_index_stats()
    print(f"\nLoaded index with {stats['total_vectors']} vectors")


if __name__ == '__main__':
    fire.Fire({
        'download': download_and_create_local_index,
        'search': search_local_index
    })