#!/usr/bin/env python3
"""
Alternative fetching approach using different Pinecone methods.
This script tries different approaches to get all 1313 vectors.
"""

import os
from pinecone import Pinecone
from tqdm import tqdm

def comprehensive_fetch():
    """Try multiple approaches to fetch all vectors."""
    print("🔄 Comprehensive Fetch Test")
    print("=" * 30)
    
    host = "https://295-vstore-51a00a2.svc.aped-4627-b74a.pinecone.io"
    index_name = "295-vstore"
    namespace = "Slides"
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not set")
        return
    
    try:
        # Connect
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name, host=host)
        
        # Get stats
        stats = index.describe_index_stats()
        expected = stats.namespaces['Slides'].vector_count if 'Slides' in stats.namespaces else 0
        print(f"Expected vectors in 'Slides': {expected}")
        
        all_ids = set()
        all_vectors = []
        all_metadata = []
        
        # Strategy 1: Systematic grid search
        print("\n📐 Strategy 1: Grid-based search")
        
        grid_size = 20  # Create a 20x20 grid of query points
        for i in tqdm(range(grid_size)):
            for j in range(grid_size):
                # Create a query vector based on grid position
                query_vector = []
                for k in range(1536):
                    if k % 2 == 0:
                        query_vector.append((i - grid_size/2) / grid_size * 0.1)
                    else:
                        query_vector.append((j - grid_size/2) / grid_size * 0.1)
                
                try:
                    result = index.query(
                        vector=query_vector,
                        top_k=100,
                        namespace=namespace,
                        include_values=True,
                        include_metadata=True
                    )
                    
                    new_count = 0
                    for match in result.matches:
                        if match.id not in all_ids:
                            all_ids.add(match.id)
                            all_vectors.append(match.values)
                            all_metadata.append(match.metadata or {})
                            new_count += 1
                    
                    if new_count > 0:
                        print(f"   Grid ({i},{j}): +{new_count} vectors (total: {len(all_vectors)})")
                    
                    if len(all_vectors) >= expected:
                        break
                
                except Exception as e:
                    continue
            
            if len(all_vectors) >= expected:
                break
        
        print(f"\nAfter grid search: {len(all_vectors)} vectors")
        
        # Strategy 2: Random sampling with high top_k
        if len(all_vectors) < expected:
            print("\n🎲 Strategy 2: High-volume random sampling")
            
            import random
            for attempt in tqdm(range(50)):
                # Generate very different random vectors
                random.seed(attempt * 123)
                query_vector = [random.uniform(-1, 1) for _ in range(1536)]
                
                try:
                    result = index.query(
                        vector=query_vector,
                        top_k=500,  # Request more vectors per query
                        namespace=namespace,
                        include_values=True,
                        include_metadata=True
                    )
                    
                    new_count = 0
                    for match in result.matches:
                        if match.id not in all_ids:
                            all_ids.add(match.id)
                            all_vectors.append(match.values)
                            all_metadata.append(match.metadata or {})
                            new_count += 1
                    
                    if new_count > 0:
                        print(f"   Random {attempt}: +{new_count} vectors (total: {len(all_vectors)})")
                    
                    if len(all_vectors) >= expected:
                        break
                
                except Exception as e:
                    continue
        
        print(f"\nFinal result: {len(all_vectors)} out of {expected} vectors")
        print(f"Coverage: {len(all_vectors)/expected*100:.1f}%")
        
        # Show sample metadata
        if all_metadata:
            print(f"\nSample metadata:")
            for i, meta in enumerate(all_metadata[:5]):
                print(f"  {i+1}. {meta}")
        
        return all_vectors, all_metadata, list(all_ids)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

if __name__ == "__main__":
    vectors, metadata, ids = comprehensive_fetch()
    
    if vectors:
        print(f"\n💾 Saving {len(vectors)} vectors to comprehensive_index.pkl")
        import pickle
        
        index_data = {
            'vectors': vectors,
            'metadata_store': {i: metadata[i] for i in range(len(metadata))},
            'vector_ids': ids,
            'vector_dim': len(vectors[0]) if vectors else 0
        }
        
        with open("comprehensive_index.pkl", 'wb') as f:
            pickle.dump(index_data, f)
        
        print("✅ Saved successfully!")
    else:
        print("❌ No vectors to save")