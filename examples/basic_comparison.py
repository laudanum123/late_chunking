"""Example script demonstrating basic RAG comparison usage."""
import asyncio
import os
import sys
from pathlib import Path
from late_chunking.rag_comparison import RAGComparison

# Sample documents about New York City
DOCUMENTS = [
    """
    New York City was founded in 1624 as a trading post by Dutch colonists. 
    Originally called New Amsterdam, it was renamed New York in 1664 after 
    the English took control. The city served as the capital of the United 
    States from 1785 to 1790.
    """,
    """
    The Empire State Building is an iconic Art Deco skyscraper in Midtown 
    Manhattan. Completed in 1931, it was the world's tallest building until 
    1970. Standing at 1,454 feet tall, it has become a symbol of New York 
    City's architectural ambition and innovation.
    """,
    """
    The New York City Subway system is one of the oldest and most extensive 
    public transit systems in the world. Opened in 1904, it now has 472 
    stations and 850 miles of track. The system operates 24/7 and serves 
    over 5.5 million riders daily.
    """
]

# Sample queries about different aspects of NYC
QUERIES = [
    "Tell me about the history of New York City's founding",
    "What are some key facts about the Empire State Building?",
    "How extensive is the NYC subway system?",
    "What makes NYC's architecture unique?",
    "When did NYC serve as the US capital?"
]

async def main():
    """Run a basic RAG comparison example."""
    try:
        # Ensure we're in the project root directory
        os.chdir(Path(__file__).parent.parent)
        
        # Initialize comparison with default config
        comparison = RAGComparison()
        
        print("\nRunning RAG comparison...")
        print("=" * 50)
        print("Queries:")
        for i, query in enumerate(QUERIES, 1):
            print(f"{i}. {query}")
        print("\nDocuments:")
        for i, doc in enumerate(DOCUMENTS, 1):
            print(f"{i}. {doc.strip()[:100]}...")
        print("=" * 50)
        
        # Run comparison
        result = await comparison.run_comparison(QUERIES, DOCUMENTS)
        
        # Print results
        print("\nRAG Comparison Results:")
        print("=" * 50)
        print(result)
        
        # Calculate average scores
        avg_traditional = sum(result.traditional_scores) / len(result.traditional_scores)
        avg_late_chunking = sum(result.late_chunking_scores) / len(result.late_chunking_scores)
        
        print("\nAverage Scores:")
        print(f"Traditional Approach: {avg_traditional:.4f}")
        print(f"Late Chunking Approach: {avg_late_chunking:.4f}")
        
        # Determine which approach performed better
        if avg_late_chunking > avg_traditional:
            print("\nLate chunking approach performed better overall!")
        elif avg_traditional > avg_late_chunking:
            print("\nTraditional approach performed better overall!")
        else:
            print("\nBoth approaches performed similarly.")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
