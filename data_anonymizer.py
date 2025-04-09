import os
from dotenv import load_dotenv
import requests
import re
from typing import List, Dict
import unicodedata
import json

# Load environment variables
load_dotenv()

class DataAnonymizer:
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        """
        Initialize the DataAnonymizer with BERT model API.
        Args:
            model_name (str): Name of the BERT model to use
        """
        self.api_key = os.getenv('HF_TOKENS')
        if not self.api_key:
            raise ValueError("HF_TOKENS not found in .env file")
            
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Dictionary to map entity labels to replacement text
        self.replacement_dict = {
            'PER': '[PERSON]',
            'ORG': '[ORGANIZATION]',
            'LOC': '[LOCATION]',
            'MISC': '[MISC]',
            'EMAIL': '[EMAIL]',
            'PHONE': '[PHONE]'
        }

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by converting to NFKC form while preserving case.
        Args:
            text (str): Input text to normalize
        Returns:
            str: Normalized text
        """
        text = unicodedata.normalize('NFKC', text)
        return text

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better entity detection while preserving original case.
        Args:
            text (str): Input text to preprocess
        Returns:
            str: Preprocessed text
        """
        # Split text into lines
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Split line into words
            words = line.split()
            processed_words = []
            
            for word in words:
                # Handle organizations and locations
                if word.isupper() and len(word) > 2:
                    # If it's a single word in uppercase, it might be an organization
                    if word.isalpha():
                        processed_words.append(word.title())
                    # If it contains numbers or special chars, it's likely an organization
                    else:
                        processed_words.append(word)
                # Handle person names
                elif word.islower() and len(word) > 2:
                    if word.isalpha() and word[0].isalpha():
                        processed_words.append(word.capitalize())
                    else:
                        processed_words.append(word)
                else:
                    processed_words.append(word)
            
            processed_lines.append(' '.join(processed_words))
        
        return '\n'.join(processed_lines)

    def _detect_email(self, text: str) -> List[Dict]:
        """
        Detect email addresses in text
        Args:
            text (str): Input text
        Returns:
            List[Dict]: List of detected email addresses with positions
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = []
        for match in re.finditer(email_pattern, text):
            matches.append({
                'entity': 'EMAIL',
                'start': match.start(),
                'end': match.end(),
                'word': match.group()
            })
        return matches

    def _detect_phone(self, text: str) -> List[Dict]:
        """
        Detect international phone numbers in text
        Args:
            text (str): Input text
        Returns:
            List[Dict]: List of detected phone numbers with positions
        """
        # International phone number patterns
        phone_patterns = [
            r'\b(?:\+?\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',  # Standard US/Canada
            r'\b(?:\+?\d{1,3}[-.]?)?\d{2,4}[-.]?\d{2,4}[-.]?\d{2,4}\b',  # International
            r'\b(?:\+?\d{1,3}[-.]?)?\d{1,4}[-.]?\d{1,4}[-.]?\d{1,4}\b',  # Flexible format
            r'\b(?:\+?\d{1,3}[-.]?)?\d{1,3}[-.]?\d{1,3}[-.]?\d{1,3}\b',  # Short format
            r'\b(?:\+?\d{1,3}[-.]?)?\d{1,2}[-.]?\d{1,2}[-.]?\d{1,2}\b'   # Very short format
        ]
        
        matches = []
        for pattern in phone_patterns:
            for match in re.finditer(pattern, text):
                # Validate the phone number
                phone = match.group().replace('-', '').replace('.', '').replace('(', '').replace(')', '')
                if len(phone) >= 6:  # Minimum length for a valid phone number
                    matches.append({
                        'entity': 'PHONE',
                        'start': match.start(),
                        'end': match.end(),
                        'word': match.group(),
                        'replacement': '[PHONE]'
                    })
        return matches

    def _query_model(self, text: str) -> List[Dict]:
        """
        Query the BERT model API for NER
        Args:
            text (str): Input text
        Returns:
            List[Dict]: NER results from model
        """
        try:
            # Preprocess text for better detection
            processed_text = self._preprocess_text(text)
            
            response = requests.post(self.api_url, headers=self.headers, json={"inputs": processed_text})
            response.raise_for_status()
            result = response.json()
            
            # Handle different possible response structures
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], list):
                    return result[0]  # Handle nested list structure
                return result  # Handle flat list structure
            elif isinstance(result, dict):
                return result.get('entities', [])  # Handle dictionary structure
            else:
                raise ValueError(f"Unexpected API response structure: {type(result)}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error querying model API: {str(e)}")

    def anonymize_text(self, text: str) -> str:
        """
        Anonymize text by detecting and replacing entities
        Args:
            text (str): Input text to anonymize
        Returns:
            str: Anonymized text
        """
        # Normalize text while preserving case
        text = self._normalize_text(text)
        
        # Get all entities from BERT model
        ner_results = self._query_model(text)
        
        # Get additional entities (email and phone)
        additional_entities = self._detect_email(text) + self._detect_phone(text)
        
        # Common location suffixes and prefixes for better detection
        location_suffixes = ('burg', 'berg', 'town', 'city', 'ville', 'polis', 'grad', 'abad', 'pur', 'nagar', 'pore', 'stan', 'land', 'ia', 'ya')
        location_prefixes = ('new', 'old', 'north', 'south', 'east', 'west', 'upper', 'lower', 'port', 'fort', 'saint', 'san', 'santa')
        
        # Common surname patterns from different cultures
        surname_patterns = {
            'slavic': ('ov', 'ev', 'ski', 'sky', 'ova', 'eva'),
            'asian': ('yan', 'ian', 'jin', 'chen', 'li', 'wang'),
            'nordic': ('sen', 'sson', 'dottir', 'dóttir'),
            'middle_eastern': ('zadeh', 'oglu', 'oglu', 'oglu', 'pour'),
            'indian': ('raj', 'kumar', 'singh', 'patel', 'sharma'),
            'hispanic': ('ez', 'es', 'os', 'as', 'is')
        }
        
        # Combine all entities and sort by start position (reversed to avoid position conflicts)
        all_entities = []
        for entity in ner_results:
            if isinstance(entity, dict):
                # Get the entity group and clean it up
                entity_group = entity.get('entity_group', entity.get('label', ''))
                # Remove any B- or I- prefix from the entity group
                entity_group = entity_group.split('-')[-1] if '-' in entity_group else entity_group
                
                start = entity.get('start', 0)
                end = entity.get('end', 0)
                word = entity.get('word', '')
                
                # Skip if the word is too short or contains special characters
                if len(word) < 2 or not word.replace('-', '').isalnum():
                    continue
                
                # Handle case sensitivity for person names
                if entity_group == 'PER':
                    # Check if the word is all uppercase
                    if word.isupper() and word.isalpha():
                        replacement = '[PERSON]'
                    # Check if the word is all lowercase
                    elif word.islower():
                        replacement = '[person]'
                    # Check if the word is title case
                    elif word.istitle():
                        # Check if it matches any surname patterns
                        is_surname = any(word.lower().endswith(pattern) for patterns in surname_patterns.values() for pattern in patterns)
                        if is_surname:
                            replacement = '[SURNAME]'
                        else:
                            replacement = '[Person]'
                    # Default to original case
                    else:
                        replacement = '[PERSON]'
                # Handle organizations
                elif entity_group == 'ORG':
                    # If it's all uppercase and contains only letters, it might be a name
                    if word.isupper() and word.isalpha():
                        replacement = '[PERSON]'
                    # If it's title case and contains only letters, it might be a name
                    elif word.istitle() and word.isalpha():
                        replacement = '[Person]'
                    # If it contains numbers or special chars, it's definitely an organization
                    elif not word.isalpha():
                        replacement = '[ORGANIZATION]'
                    else:
                        replacement = self.replacement_dict.get(entity_group, '[UNKNOWN]')
                # Handle locations
                elif entity_group == 'LOC':
                    # If it's all uppercase and contains only letters, it might be a name
                    if word.isupper() and word.isalpha():
                        replacement = '[PERSON]'
                    # If it's title case and contains only letters, it might be a name
                    elif word.istitle() and word.isalpha():
                        # Check if it's likely a location based on suffixes/prefixes
                        if word.lower().endswith(location_suffixes) or word.lower().startswith(location_prefixes):
                            replacement = '[LOCATION]'
                        else:
                            replacement = '[Person]'
                    else:
                        replacement = self.replacement_dict.get(entity_group, '[UNKNOWN]')
                else:
                    replacement = self.replacement_dict.get(entity_group, '[UNKNOWN]')
                
                all_entities.append({
                    'entity': entity_group,
                    'start': start,
                    'end': end,
                    'word': word,
                    'replacement': replacement
                })
        
        all_entities.extend(additional_entities)
        all_entities.sort(key=lambda x: x['start'], reverse=True)
        
        # Replace entities with anonymized versions
        anonymized_text = text
        for entity in all_entities:
            replacement = entity.get('replacement', self.replacement_dict.get(entity['entity'], '[UNKNOWN]'))
            
            # Get the word boundaries
            start = entity['start']
            end = entity['end']
            
            # Find the complete word boundaries
            while start > 0 and anonymized_text[start-1].isalnum():
                start -= 1
            while end < len(anonymized_text) and anonymized_text[end].isalnum():
                end += 1
            
            # Only replace if we're at word boundaries and the word is complete
            if (start == 0 or not anonymized_text[start-1].isalnum()) and \
               (end == len(anonymized_text) or not anonymized_text[end].isalnum()):
                # Check if we're replacing a complete word
                word_to_replace = anonymized_text[start:end]
                if word_to_replace.isalnum() and len(word_to_replace) > 1:
                    anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
            
        return anonymized_text

# Example usage
if __name__ == "__main__":
    # Create an instance of DataAnonymizer
    anonymizer = DataAnonymizer()
    
    # Test with mixed case text containing various entities
    sample_text = """
    John Smith works at Microsoft Corporation in New York City.
    You can reach him at john.smith@microsoft.com or call +1-555-123-4567.
    """
    
    try:
        # Get the entities from BERT model
        ner_results = anonymizer._query_model(sample_text)
        
        # Get additional entities (email and phone)
        additional_entities = anonymizer._detect_email(sample_text) + anonymizer._detect_phone(sample_text)
        
        # Combine and sort all entities
        all_entities = []
        for entity in ner_results:
            if isinstance(entity, dict):
                # Get the entity group and clean it up
                entity_group = entity.get('entity_group', entity.get('label', ''))
                # Remove any B- or I- prefix from the entity group
                entity_group = entity_group.split('-')[-1] if '-' in entity_group else entity_group
                
                start = entity.get('start', 0)
                end = entity.get('end', 0)
                word = entity.get('word', '')
                
                all_entities.append({
                    'entity': entity_group,
                    'start': start,
                    'end': end,
                    'word': word
                })
        
        all_entities.extend(additional_entities)
        all_entities.sort(key=lambda x: x['start'], reverse=True)  # Sort in reverse for proper replacement
        
        print("Original Text:")
        print(sample_text)
        print("\nDetected Entities:")
        for entity in sorted(all_entities, key=lambda x: x['start']):  # Sort forward for display
            print(f"Text: '{entity['word']}' -> Label: {entity['entity']}")
        
        # Create anonymized text
        anonymized_text = sample_text
        for entity in all_entities:
            replacement = anonymizer.replacement_dict.get(entity['entity'], '[UNKNOWN]')
            anonymized_text = anonymized_text[:entity['start']] + replacement + anonymized_text[entity['end']:]
        
        print("\nAnonymized Text:")
        print(anonymized_text)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

