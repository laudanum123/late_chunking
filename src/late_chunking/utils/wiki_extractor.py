import wikipediaapi
import argparse
from urllib.parse import urlparse, unquote
import os

def extract_page_title_from_url(url: str) -> str:
    """
    Extract the page title from a Wikipedia URL.
    
    Args:
        url (str): Wikipedia URL
        
    Returns:
        str: Page title
    """
    # Parse the URL
    parsed = urlparse(url)
    
    # Check if it's a Wikipedia URL
    if 'wikipedia.org' not in parsed.netloc:
        raise ValueError("Not a Wikipedia URL")
    
    # Get the page title from the path
    # The path is usually in the format: /wiki/Page_Title
    path_parts = parsed.path.split('/')
    if len(path_parts) < 3 or path_parts[1] != 'wiki':
        raise ValueError("Invalid Wikipedia URL format")
    
    # Get the page title and decode URL encoding
    page_title = unquote(path_parts[2])
    # Replace underscores with spaces
    page_title = page_title.replace('_', ' ')
    
    return page_title

def get_wiki_content(url: str, output_dir: str = "documents") -> str:
    """
    Extract content from a Wikipedia article URL and save it to a file.
    
    Args:
        url (str): Wikipedia article URL
        output_dir (str): Directory to save the output file
        
    Returns:
        str: Path to the saved file
    """
    # Extract page title from URL
    try:
        page_title = extract_page_title_from_url(url)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return None
    
    # Initialize Wikipedia API
    wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='LateChunkingRAG/1.0 (https://github.com/your-username/late_chunking; your-email@example.com)'
    )
    
    # Get the page
    page = wiki.page(page_title)
    
    # Check if page exists
    if not page.exists():
        print(f"Error: Page '{page_title}' does not exist")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from page title
    filename = page_title.replace(' ', '_').lower() + '.txt'
    filepath = os.path.join(output_dir, filename)
    
    # Extract and save the content
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write title
            f.write(f"{page.title}\n\n")
            
            # Write summary
            if page.summary:
                f.write("Summary:\n")
                f.write(page.summary)
                f.write("\n\n")
            
            # Write full text
            f.write("Full Article:\n")
            f.write(page.text)
        
        print(f"Successfully saved article to: {filepath}")
        return filepath
    
    except Exception as e:
        print(f"Error saving content: {str(e)}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract content from Wikipedia articles')
    parser.add_argument('urls', nargs='+', help='Wikipedia article URLs')
    parser.add_argument('--output-dir', default='documents',
                      help='Directory to save the extracted articles (default: documents)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process each URL
    for url in args.urls:
        print(f"\nProcessing: {url}")
        filepath = get_wiki_content(url, args.output_dir)
        if filepath:
            print(f"Content saved to: {filepath}")
        print("-" * 80)

if __name__ == "__main__":
    main()
