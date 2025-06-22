import re

class TextCleaner:
    def __init__(self):
        """
        Initializes the TextCleaner.
        """
        # Regex to find a word ending in a hyphen, followed by a newline,
        # and then a word starting with a lowercase letter.
        self.dehyphenate_regex = re.compile(r'(\w+)-\n(\w+)')
        
        # Regex to find a newline that is NOT preceded by sentence-ending punctuation.
        # This is used to "unwrap" paragraphs by joining lines that shouldn't be broken.
        self.dewrap_regex = re.compile(r'(?<![.!?:"])\s*\n(?![\n\s])')

    def clean_text(self, text: str) -> str:
        """
        Applies a series of cleanup steps to the extracted text.
        
        Args:
            text (str): The raw extracted text.
            
        Returns:
            str: The cleaned text.
        """
        if not text or not isinstance(text, str):
            return ""
            
        # The order of these operations can be important
        text = self.dehyphenate_text(text)
        text = self.dewrap_text(text)
        text = self.normalize_whitespace(text)
        return text

    def dehyphenate_text(self, text: str) -> str:
        """
        Finds hyphenated words at the end of lines and joins them.
        Example: "This is a Revolution-\nary idea." -> "This is a Revolutionary idea."
        """
        return self.dehyphenate_regex.sub(r'\1\2', text)

    def dewrap_text(self, text: str) -> str:
        """
        Joins lines that are part of the same paragraph by replacing single
        newlines with spaces, preserving paragraph breaks (double newlines).
        """
        return self.dewrap_regex.sub(' ', text)

    def normalize_whitespace(self, text: str) -> str:
        """
        Replaces multiple whitespace characters with a single space
        and tidies up spaces around newlines.
        """
        # Replace multiple spaces/tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Tidy up spaces around newlines
        text = re.sub(r' \n', '\n', text)
        text = re.sub(r'\n ', '\n', text)
        # Replace 3 or more newlines with just two (to preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip() 