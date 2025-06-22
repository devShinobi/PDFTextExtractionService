import sys
import os
import pytest

# Add project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_cleaner import TextCleaner
from utils.context_analyzer import ContextAnalyzer

# --- TextCleaner Tests ---

@pytest.fixture
def cleaner():
    return TextCleaner()

def test_dehyphenation(cleaner):
    text = "This is a revolu-\ntionary idea that will hopefully trans-\nform the world."
    expected = "This is a revolutionary idea that will hopefully transform the world."
    assert cleaner.dehyphenate_text(text) == expected

def test_paragraph_dewrapping(cleaner):
    text = "This is the first line of a paragraph.\nThis is the second line, which should be joined.\n\nThis is a new paragraph that should remain separate."
    expected = "This is the first line of a paragraph. This is the second line, which should be joined.\n\nThis is a new paragraph that should remain separate."
    assert cleaner.dewrap_text(text) == expected

def test_whitespace_normalization(cleaner):
    text = "  This   has    too much \n\n\n   whitespace.  "
    expected = "This has too much\n\nwhitespace."
    assert cleaner.normalize_whitespace(text) == expected

def test_full_cleanup(cleaner):
    raw_text = """
    This is an example of a para-\ngraph that needs a lot of work.
    It has poor line breaks, and     extra spaces.

    Hopefully, the cleaner can fix it all up nicely.
    """
    cleaned_text = "This is an example of a paragraph that needs a lot of work. It has poor line breaks, and extra spaces.\n\nHopefully, the cleaner can fix it all up nicely."
    assert cleaner.clean_text(raw_text) == cleaned_text

# --- ContextAnalyzer Tests ---

@pytest.fixture
def analyzer():
    # Using a smaller, faster model for testing if available, but MiniLM is robust.
    return ContextAnalyzer(model_name='all-MiniLM-L6-v2')

def test_context_outlier_detection(analyzer):
    """
    Tests the core outlier detection logic.
    - Paragraphs 0, 1, 3 are about technology.
    - Paragraph 2 is about fruit.
    - Paragraph 4 is a short, unrelated command.
    """
    if not analyzer.model:
        pytest.skip("SentenceTransformer model not available.")

    paragraphs = [
        "The latest advancements in AI are revolutionizing the tech industry.",
        "Machine learning models now power many of our daily applications.",
        "An apple a day keeps the doctor away.", # Outlier
        "Quantum computing promises to solve currently intractable problems.",
        "Please confirm your subscription.", # Outlier
    ]
    
    outlier_indices = analyzer.find_outliers(paragraphs, eps=0.6)
    
    # The exact indices depend on the clustering, but we expect 2 outliers.
    # We expect 'apple' and 'subscription' to be outliers.
    assert len(outlier_indices) == 2, "Should detect the correct number of outliers"
    assert 2 in outlier_indices, "The 'apple' paragraph should be an outlier"
    assert 4 in outlier_indices, "The 'subscription' paragraph should be an outlier"

def test_not_enough_text_for_analysis(analyzer):
    """
    Tests that the analyzer gracefully handles cases with too little text.
    """
    if not analyzer.model:
        pytest.skip("SentenceTransformer model not available.")
        
    paragraphs = ["This is one sentence.", "This is another."]
    outlier_indices = analyzer.find_outliers(paragraphs, min_samples=2)
    assert outlier_indices == [], "Should return no outliers if text is insufficient"

if __name__ == "__main__":
    pytest.main() 