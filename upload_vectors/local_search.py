"""
Enhanced local search functionality with OpenAI embeddings integration.
This script provides a complete solution for searching your local vector index
using the same embedding model as your upload process.
"""

import os
import json
# import numpy as np  # Removed to avoid multiarray issues
from typing import List, Dict, Any, Optional
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# import faiss  # Removed to avoid numpy dependency
import pickle
from pinecone_to_local import PineconeToLocal
import fire


class LocalVectorSearch:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the local vector search with OpenAI embeddings.
        
        Args:
            openai_api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize embeddings (same model as used in image_input.py)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key, 
            model="text-embedding-ada-002"
        )
        
        self.converter = PineconeToLocal()
        self.local_index = None
        self.metadata_store = {}
        self.vector_ids = []
    
    def load_index(self, index_path: str) -> bool:
        """Load the local FAISS index."""
        return self.converter.load_local_index(index_path)
    
    def embed_text(self, text: str) -> List[float]:
        """Convert text to embedding vector using OpenAI."""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"Error embedding text: {e}")
            return []
    
    def search_by_text(self, query_text: str, top_k: int = 10, 
                      file_filter: str = None, slide_filter: str = None) -> List[Dict]:
        """
        Search the local index using text query.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            file_filter: Filter results by file name (partial match)
            slide_filter: Filter results by slide number
            
        Returns:
            List of search results with metadata
        """
        if not self.converter.local_index:
            print("Local index not loaded. Use load_index() first.")
            return []
        
        # Convert text to vector
        query_vector = self.embed_text(query_text)
        if not query_vector:
            return []
        
        # Search the index
        results = self.converter.search_local_index(query_vector, top_k=top_k * 2)  # Get more for filtering
        
        # Apply filters
        filtered_results = []
        for result in results:
            metadata = result.get('metadata', {})
            
            # File filter
            if file_filter and file_filter.lower() not in metadata.get('file', '').lower():
                continue
            
            # Slide filter
            if slide_filter and slide_filter != str(metadata.get('Slide', '')):
                continue
            
            filtered_results.append(result)
            
            # Stop when we have enough results
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    def search_by_vector(self, query_vector: List[float], top_k: int = 10) -> List[Dict]:
        """Search using a pre-computed vector."""
        if not self.converter.local_index:
            print("Local index not loaded. Use load_index() first.")
            return []
        
        return self.converter.search_local_index(query_vector, top_k)
    
    def get_all_files(self) -> List[str]:
        """Get list of all unique files in the index."""
        if not self.converter.metadata_store:
            return []
        
        files = set()
        for metadata in self.converter.metadata_store.values():
            if 'file' in metadata:
                files.add(metadata['file'])
        
        return sorted(list(files))
    
    def get_slides_for_file(self, file_name: str) -> List[str]:
        """Get all slide numbers for a specific file."""
        if not self.converter.metadata_store:
            return []
        
        slides = set()
        for metadata in self.converter.metadata_store.values():
            if metadata.get('file', '').endswith(file_name):
                if 'Slide' in metadata:
                    slides.add(str(metadata['Slide']))
        
        return sorted(list(slides))
    
    def print_search_results(self, results: List[Dict], show_description: bool = True):
        """Pretty print search results."""
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} results:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            score = result.get('score', 0)
            
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   File: {metadata.get('file', 'Unknown')}")
            print(f"   Slide: {metadata.get('Slide', 'Unknown')}")
            print(f"   ID: {result.get('id', 'Unknown')}")
            
            if show_description and 'description' in metadata:
                description = metadata['description']
                # Truncate long descriptions
                if len(description) > 200:
                    description = description[:200] + "..."
                print(f"   Description: {description}")
            
            print("-" * 40)


def setup_local_index(
    index_name: str = "295-youtube-index",
    namespace: str = "Slides",
    output_path: str = "local_index",
    top_k: int = 10000,
    host: str = "https://295-vstore-51a00a2.svc.aped-4627-b74a.pinecone.io"
):
    """
    Download vectors from Pinecone and create a local index.
    
    Usage:
        python local_search.py setup_local_index
        python local_search.py setup_local_index --namespace="Slides" --output_path="my_index"
    """
    from pinecone_to_local import download_and_create_local_index
    
    print("Setting up local index from Pinecone...")
    success = download_and_create_local_index(
        index_name=index_name,
        namespace=namespace,
        output_path=output_path,
        top_k=top_k,
        host=host
    )
    
    if success:
        print(f"\n✅ Local index successfully created at: {output_path}")
        print(f"You can now search using: python local_search.py search 'your query'")
    else:
        print("❌ Failed to create local index")


def search(
    query: str,
    index_path: str = "local_index",
    top_k: int = 5,
    file_filter: str = None,
    slide_filter: str = None,
    show_description: bool = True
):
    """
    Search the local vector index with a text query.
    
    Usage:
        python local_search.py search "machine learning concepts"
        python local_search.py search "pipeline hazards" --top_k=10
        python local_search.py search "cache" --file_filter="L29"
        python local_search.py search "memory" --slide_filter="05"
    """
    try:
        searcher = LocalVectorSearch()
        
        print(f"Loading index from: {index_path}")
        if not searcher.load_index(index_path):
            print(f"❌ Failed to load index from {index_path}")
            print(f"Make sure to run: python local_search.py setup_local_index")
            return
        
        print(f"🔍 Searching for: '{query}'")
        if file_filter:
            print(f"   📁 File filter: {file_filter}")
        if slide_filter:
            print(f"   📄 Slide filter: {slide_filter}")
        
        results = searcher.search_by_text(
            query_text=query,
            top_k=top_k,
            file_filter=file_filter,
            slide_filter=slide_filter
        )
        
        searcher.print_search_results(results, show_description=show_description)
        
    except Exception as e:
        print(f"❌ Error during search: {e}")


def list_files(index_path: str = "local_index"):
    """
    List all files in the index.
    
    Usage:
        python local_search.py list_files
    """
    try:
        searcher = LocalVectorSearch()
        
        if not searcher.load_index(index_path):
            print(f"❌ Failed to load index from {index_path}")
            return
        
        files = searcher.get_all_files()
        print(f"\n📁 Found {len(files)} files in the index:")
        print("=" * 50)
        
        for i, file in enumerate(files, 1):
            slides = searcher.get_slides_for_file(file.split('/')[-1])
            print(f"{i:2d}. {file}")
            print(f"     Slides: {len(slides)} ({', '.join(slides[:10])}{'...' if len(slides) > 10 else ''})")
        
    except Exception as e:
        print(f"❌ Error listing files: {e}")


def index_stats(index_path: str = "local_index"):
    """
    Show statistics about the local index.
    
    Usage:
        python local_search.py index_stats
    """
    try:
        searcher = LocalVectorSearch()
        
        if not searcher.load_index(index_path):
            print(f"❌ Failed to load index from {index_path}")
            return
        
        stats = searcher.converter.get_index_stats()
        
        print("\n📊 Index Statistics:")
        print("=" * 30)
        print(f"Total vectors: {stats['total_vectors']}")
        print(f"Vector dimension: {stats['vector_dimension']}")
        print(f"Index type: {stats['index_type']}")
        print(f"Metadata entries: {stats['metadata_entries']}")
        
        # Count unique files
        files = searcher.get_all_files()
        print(f"Unique files: {len(files)}")
        
        # Show sample metadata
        if stats['sample_metadata']:
            print(f"\n📄 Sample entries:")
            for i, sample in enumerate(stats['sample_metadata'][:3]):
                file_name = sample.get('file', 'Unknown').split('/')[-1]
                slide = sample.get('Slide', 'Unknown')
                description = sample.get('description', '')[:100] + "..." if len(sample.get('description', '')) > 100 else sample.get('description', '')
                print(f"  {i+1}. {file_name} - Slide {slide}")
                print(f"     {description}")
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")


if __name__ == '__main__':
    fire.Fire({
        'setup_local_index': setup_local_index,
        'search': search,
        'list_files': list_files,
        'index_stats': index_stats
    })