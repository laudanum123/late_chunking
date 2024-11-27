"""Comprehensive RAG comparison example using FixedTokenChunker."""
import asyncio
import os
from pathlib import Path
import numpy as np
from typing import List, Dict
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from late_chunking.rag_comparison import RAGComparison
from late_chunking.chunkers import FixedTokenChunker
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample questions about the cities
QUESTIONS = [
    'Kann man eine Briefmarke selbst gestalten?',
    'Wie können Unterrichtsmaterialien bestellt werden?',
    'Darf ich Medikamente per Brief versenden?',
    'Ich habe mich bei Ihnen beworben. Ich habe immer noch keine Antwort erhalten',
    'Was mache ich mit einer Sponsoring Anfrage?',
    'Kostet GoGreen für Pakete extra?',
    'Kann eine Privatperson ein Brief mit Nachnahme versenden?',
    'Ich wollte mich über eine Filiale beschweren. Die Mitarbeiter waren alle sehr unfreundlich und haben mir nicht weitergeholfen.',
    'Kann ich mein Päkchen und die Frankierung online bezahlen?',
    'Warum kann ich meine Sendung nicht umleiten?',
    'Ich kann aus gesundheitlichen Gründen das Paket aus der Packstation nicht abholen',
    'Warum wurde mein Paket nicht zugestellt, ich war den ganzen Tag zuhause?',
    'Ich habe eine kostenlose Abholung gebucht. Der Zusteller hat sich aber geweigert das Paket mitzunehmen.',
    'Kann ich ein Paket als Retoure im Paket-Shop abgeben?',
    'Ich habe mein Passwort für das Kundenkonto vergessen. Was muss ich tun?',
    'Wieso ist mein Brief unzustellbar?',
    'Der Zusteller hat mit dem Auto meine Hauswand beschädigt.',
    'Ich möchte mein Paket immer auf meine Terasse geliefert bekommen.',
    'Ich habe eine Benachrichtigung bekommen das mein Paket bald ankommt. Ich habe aber nichts bestellt.',
    'Muss der Name am Briefkasten oder an der Klingel stehen?',
    'Kann der Zusteller Sendungen in eine RENZbox einstellen?',
    'Kann man ein Päckchen in den Briefkasten werfen?',
    'Kann man eine Urne versenden?',
    'Der Inhalt von meinem Paket ist zerstört. Was passiert jetzt?',
    'Wo kann man einen Bahntransport buchen?',
    'Wie kann man den Lagerservice beauftragen?',
    'Wie funktiniert der Lagerservice?',
    'Kann man 5 Euro für die Portokasse aufladen?',
    'Ich habe geheiratet. Wo kann ich meinen Namen ändern?',
    'Ich möchte eine 2F Authentifizierung in meinem Kundenkonto',
    'Kunde soll über SMS 1,99 EUR Zoll bezahlen - warum?',
    'Die DHL App funktioniert nicht und zeigt an ""App angehalten"".',
    'Wie kann ich ein aktiviertes Gerät entfernen?',
    'Die Geräteaktivierung funktioniert nicht.',
    'Kunde möchte keine digitale Benachrichtigung mehr.',
    'Was tun wenn eine Briefsendung beschädigt wurde?',
    'Paketsendung wurde beschädigt',
    'Sendung wurde in die Packstation fehlerhaft eingestellt.',
    'Was bedeutet ""nicht AGB konform""?',
    'Wie fülle ich in MySMF das Feld ""Melder"" aus?',
    'Kann man eine Ersatzzustellung an Nachbarn ausschließen?',
    'Bekomme ich einen Rabatt bei einer Online Briefmarke?',
    'Ich möchte mich über meinen Postboten beschweren.',
    'Was mache ich wenn sich ein Anwalt für einen Kunden meldet?',
    'Kann ein Paket in Folie umwickelt versendet werden?',
    'Der Zusteller hat den Kunden beleidigt.',
    'Wie lange dauert es noch bis mein Brief ankommt?',
    'Wie kann ich eine kostenlose Paketmitnahme buchen?',
    'Kann ich den Liefertag ändern?',
    'Wie kann eine Firma einen Ablagevertrag hinterlegen?',
    'Ab wie viel Euro Warenwert muss ich Zollgebühren bezahlen?',
    'Kann der Empfänger eine Nachforschung zu einer Warenpost Sendung einleiten?',
    'Ich möchte ein Brief per Einschreiben versenden, wie teuer ist das?',
    'Ich möchte mein DHL Kundenkonto kündigen, wie funktioniert das?',
    'Kann ich einen Familienangehörigen im Kundenkonto für einen Ablagevertrag eintragen?',
    'Was ist ein Familienkonto?',
    'Mein Vater ist verstorben, was passiert mit seiner Post?',
    'Wie kann ich die Annahme einer Sendung verweigern?',
    'Was mache ich wenn Sendung nur elektronisch angekündigt wurde?',
    'Was sind PAN Daten?',
    'Kann man ein Jagdgewehr versenden?',
    'Warum geht meine Sendung zurück an den Absender?',
    'Wie lange dauert der Versand eines Paket wenn der Service Bahntransport mit beauftragt wird?',
    'Die Packstation am Münsterplatz ist total mit Graffiti versaut. Kann man das nicht mal sauber machen?',
    'Briefkasten am Münsterplatz in Bonn ist total verdreckt. Kunde wünscht Reinigung',
    'Ich habe versehentlich meinen Schlüssel beim Einwurf meiner Briefe in den Briefkasten mit in diesen geworfen. Ich möchte meinen Schlüssel sofort wiederhaben.',
    'Paket wurde an Betrüger geschickt. Wie kann die Zustellung verhindert werden?',
    'Kunde hat selbst einen Nachsendeauftrag, erhält aber darüber Sendungen einer anderen Person',
    'Kunde erhält Sendungen mit Aufkleber „kein aktiver Auftrag vorhanden“',
    'Was muss ich tun, damit meine Pakete immer bei meiner Nachbarin in Haus abgegeben werden? Die ist immer zuhause.',
    'Wie kann ich meinen Zusteller erreichen? Ich möchte ihm was wegen der Zustellung sagen.',
    'Wie hoch ist der Zuschlag, für ein Paket das ich als Sperrgut versendet wird?',
    'Wie lange lange wäre mein Paket von Hamburg nach München unterwegs?',
    'Ist es möglich bei uns auf dem Firmengelände eine Packstation aufzustellen? Wir haben sehr viele Mitarbeiter und Pakete die diese sich an die Firmenadresse senden lassen?',
    'Ich möchte mein Einschreiben abholen. Jetzt habe ich gehört ich muss mich ausweisen dafür. Reicht mein Führerscheindafür?',
    'Ich möchte mein Paket abholen. Jetzt habe ich gehört ich muss mich ausweisen dafür. Reicht dafür mein Führerschein?',
    'Ich möchte ein großes Paket ins Ausland versenden',
    'Ich möchte meinen Ablagevertrag kündigen, wie funktioniert das?',
    'Ich habe meine Postcard verloren',
    'Ich möchte in meinem Laden ein DHL Paket Shop betreiben',
    'Ich möchte bei der Post oder der DHL arbeiten',
    'Ich möchte eine Transportversicherung abschließen',
    'Meine Sendung wurde in eine zu weit entfernte Packstation eingestellt',
    'Mein Paket befindet sich in einer Filiale die zu weit weg ist.',
    'Ich möchte einen Brief an meinem Mann schreiben, er ist Soldat in einem Kriegsgebiet. Was muss beachtet werden?',
    'Ich wollte ein benachrichtigtes Einschreiben für meine Schwester in der Filiale abholen. Die sagten mir dort ich brauche eine Vollmacht von ihr. Wie soll diese aussehen?',
    'Ich wollte ein Paket für meine Schwester in der Filiale abholen. Die sagten mir dort ich brauche eine Vollmacht von ihr. Wie soll diese aussehen?',
    'Kann ich mir einen Brief direkt an eine Filiale schicken lassen?',
    'Wie muss ich ein Paket beschriften, wenn ich es direkt an eine Filiale schicken möchte?',
    'Fach klemt ich konnte die Pakete nicht entnehmen',
    'Packstation hat sich nicht geöffnet mein Paket liegt in Fach',
    'wie kan ich eine Vollmacht austellen',
    'ich möchte meine Kosten wegen de beschädigten Inhalt der Sendung erstattet haben',
    'Ich habe von Amazon ein Paket bekommen und musste Nachentgelt bezahlen. Warum? Ich möchte mein Geld zurück.',
    'ich habe keine Benachrichtigungs karte erhalten wie soll ich mein Paket abholen (Packstation)',
    'ich habe keine Benachrichtigungs karte erhalten wie soll ich mein Paket abholen (Filiale)',
    'Wie lange dauert eine Nach Forschungsauftrag',
    'Empfanger ist nicht zu ermitteln',
    'Wie kan ich eine Abholung buchen',
    'Wie kan ich eine Paket im Packstation bekomen',
    'Die Sendung liegt nicht in der Packstation',
    'Die Sendung wurde an den Falschen Empfänger zugestellt',
    'Zusteller hatte nicht geklingelt',
    'Was muss der Kunde tun, damit ein Paket beim Nachbar abgeben werden kann.',
    'Wie kan ich jemanden bevollmächtigen und eine Sendung abzuholen',
    'kan ich die lagger frist in Packstation verlängern',
    'kan ich die lager frist in Filiale verlängern',
]

def load_documents() -> Dict[str, str]:
    """Load documents from the documents directory.
    
    Returns:
        Dict mapping document names to their contents
    """
    docs_dir = Path("src/late_chunking/data/documents")
    documents = {}
    
    for doc_file in docs_dir.glob("*.txt"):
        with open(doc_file, "r", encoding="utf-8") as f:
            documents[doc_file.stem] = f.read()
    
    return documents

def visualize_embeddings(embeddings: np.ndarray, doc_ids: List[str], title: str, output_file: str):
    """
    Create a 2D visualization of embeddings using t-SNE.
    
    Args:
        embeddings (np.ndarray): Matrix of embeddings
        doc_ids (List[str]): List of document IDs corresponding to each embedding
        title (str): Title for the plot
        output_file (str): Path to save the visualization
    """
    # Create t-SNE reducer
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    
    # Reduce dimensionality to 2D
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Create a mapping of unique document IDs to colors
    unique_docs = list(set(doc_ids))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_docs)))
    doc_to_color = dict(zip(unique_docs, colors))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot points colored by document
    for doc_id in unique_docs:
        mask = [d == doc_id for d in doc_ids]
        points = reduced_embeddings[mask]
        plt.scatter(points[:, 0], points[:, 1], c=[doc_to_color[doc_id]], alpha=0.6)
    
    plt.title(title)
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()

async def main():
    """Run comprehensive RAG comparison with FixedTokenChunker."""
    try:
        # Create vector store directories
        vector_store_dir = Path("outputs/fixed_token_comparison/vector_stores")
        vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Load documents
        documents = load_documents()
        logger.info(f"Loaded {len(documents)} documents")
        
        # Initialize comparison with FixedTokenChunker
        async with RAGComparison() as comparison:
            # Configure chunkers
            tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-de")
            fixed_token_chunker = FixedTokenChunker(tokenizer=tokenizer, chunk_size=256, overlap=20)  # Using smaller chunks with overlap
            comparison.late_embedder.chunker = fixed_token_chunker
            comparison.trad_embedder.chunker = fixed_token_chunker
            
            # Set vector store paths for embedders
            comparison.late_embedder.set_vector_store_path(vector_store_dir / "late_chunking")
            comparison.trad_embedder.set_vector_store_path(vector_store_dir / "traditional")
            
            # Try to load existing vector stores
            late_store_loaded = comparison.late_embedder._load_vector_store()
            trad_store_loaded = comparison.trad_embedder._load_vector_store()
            if late_store_loaded and trad_store_loaded:
                logger.info("Successfully loaded existing vector stores")
                # Extract embeddings from loaded stores
                trad_embeddings = comparison.trad_embedder.chunks
                late_embeddings = comparison.late_embedder.chunks
            else:
                logger.info("Computing new embeddings...")
                # Process late chunking first
                logger.info("Computing late chunking embeddings...")
                late_embeddings = await comparison._embed_documents_late(
                    list(documents.values()),
                    list(documents.keys())  # Use document names as IDs
                )
                # Save late chunking embeddings immediately
                logger.info("Saving late chunking vector store...")
                # Convert embeddings to numpy array and verify shape
                late_embeddings_array = np.array([chunk.embedding for chunk in late_embeddings])
                if len(late_embeddings) != late_embeddings_array.shape[0]:
                    raise ValueError(f"Mismatch between number of chunks ({len(late_embeddings)}) and embeddings ({late_embeddings_array.shape[0]})")
                
                # Reset embedder state
                comparison.late_embedder.chunks = []
                comparison.late_embedder.index = None
                
                # Add and save embeddings
                comparison.late_embedder._add_embeddings(late_embeddings_array, late_embeddings)
                comparison.late_embedder._save_vector_store()
                logger.info(f"Late chunking vector store saved successfully with {len(late_embeddings)} chunks")
                
                # Then process traditional approach
                logger.info("Computing traditional embeddings...")
                trad_embeddings = await comparison._embed_documents_traditional(
                    list(documents.values()),
                    list(documents.keys())  # Use document names as IDs
                )
                
                # Save traditional embeddings immediately
                logger.info("Saving traditional vector store...")
                # Convert embeddings to numpy array and verify shape
                trad_embeddings_array = np.array([chunk.embedding for chunk in trad_embeddings])
                if len(trad_embeddings) != trad_embeddings_array.shape[0]:
                    raise ValueError(f"Mismatch between number of chunks ({len(trad_embeddings)}) and embeddings ({trad_embeddings_array.shape[0]})")
                
                # Reset embedder state
                comparison.trad_embedder.chunks = []
                comparison.trad_embedder.index = None
                
                # Add and save embeddings
                comparison.trad_embedder._add_embeddings(trad_embeddings_array, trad_embeddings)
                comparison.trad_embedder._save_vector_store()
                logger.info(f"Traditional vector store saved successfully with {len(trad_embeddings)} chunks")
            
            # Create output directory for detailed results
            output_dir = Path("outputs/fixed_token_comparison/results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all questions
            results = []
            for question in QUESTIONS:
                result = await comparison._process_query(question, trad_embeddings, late_embeddings)
                results.append(result)

            # Save results to file
            output_path = output_dir / "comparison_results.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(f"{result}\n")
            logger.info(f"Results saved to {output_path}")

            # Create visualizations
            def get_embeddings_and_ids(chunks):
                """Extract embeddings and doc_ids from chunks, handling both ChunkWithEmbedding and dict formats."""
                embeddings = []
                doc_ids = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        embeddings.append(chunk['embedding'])
                        doc_ids.append(chunk.get('doc_id', 'unknown'))
                    elif hasattr(chunk, 'embedding') and hasattr(chunk, 'doc_id'):
                        embeddings.append(chunk.embedding)
                        doc_ids.append(chunk.doc_id)
                    else:
                        logger.warning(f"Skipping chunk without proper attributes: {chunk}")
                return np.array(embeddings), doc_ids

            # Process traditional embeddings
            trad_embeddings_array, trad_doc_ids = get_embeddings_and_ids(trad_embeddings)
            if len(trad_embeddings_array) > 0:
                visualize_embeddings(
                    trad_embeddings_array,
                    trad_doc_ids,
                    "Traditional Chunking Embeddings",
                    str(output_dir / "traditional_embeddings.png")
                )
                logger.info("Traditional embeddings visualization created")

            # Process late chunking embeddings
            late_embeddings_array, late_doc_ids = get_embeddings_and_ids(late_embeddings)
            if len(late_embeddings_array) > 0:
                visualize_embeddings(
                    late_embeddings_array,
                    late_doc_ids,
                    "Late Chunking Embeddings",
                    str(output_dir / "late_chunking_embeddings.png")
                )
                logger.info("Late chunking embeddings visualization created")
            
            return 0
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except Exception as e:
        import traceback
        logger.error(f"Error in main: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit(130)
