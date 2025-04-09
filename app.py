import streamlit as st
from data_anonymizer import DataAnonymizer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Text Anonymizer",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stTextArea {
        font-size: 16px;
    }
    .main {
        padding: 2rem;
    }
    .stMarkdown {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Title and description
    st.title("Text Anonymizer")
    st.markdown("""
    This app helps you anonymize sensitive information in text by detecting and replacing:
    - Person names
    - Organizations
    - Locations
    - Email addresses
    - Phone numbers
    """)

    # Initialize the anonymizer
    try:
        anonymizer = DataAnonymizer()
    except ValueError as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure you have set the HF_TOKENS environment variable in your .env file")
        return

    # Create two columns for input and output
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Text")
        # Text input area
        input_text = st.text_area(
            "Enter the text you want to anonymize:",
            height=300,
            placeholder="Enter your text here... (Names will be replaced with [PERSON], [person], or [Person] based on their case)"
        )

        # Anonymize button
        if st.button("Anonymize Text", type="primary"):
            if input_text.strip():
                try:
                    # Get the entities from BERT model
                    ner_results = anonymizer._query_model(input_text)
                    
                    # Get additional entities (email and phone)
                    additional_entities = anonymizer._detect_email(input_text) + anonymizer._detect_phone(input_text)
                    
                    # Add replacement information to additional entities
                    for entity in additional_entities:
                        entity['replacement'] = anonymizer.replacement_dict.get(entity['entity'], '[UNKNOWN]')
                    
                    # Combine and sort all entities
                    all_entities = []
                    for entity in ner_results:
                        if isinstance(entity, dict):
                            entity_group = entity.get('entity_group', entity.get('label', ''))
                            entity_group = entity_group.split('-')[-1] if '-' in entity_group else entity_group
                            
                            start = entity.get('start', 0)
                            end = entity.get('end', 0)
                            word = entity.get('word', '')
                            
                            # Handle case sensitivity for person names
                            if entity_group == 'PER':
                                # Check if the word is all uppercase
                                if word.isupper():
                                    replacement = '[PERSON]'
                                # Check if the word is all lowercase
                                elif word.islower():
                                    replacement = '[person]'
                                # Check if the word is title case
                                elif word.istitle():
                                    replacement = '[Person]'
                                # Default to original case
                                else:
                                    replacement = '[PERSON]'
                            else:
                                replacement = anonymizer.replacement_dict.get(entity_group, '[UNKNOWN]')
                            
                            all_entities.append({
                                'entity': entity_group,
                                'start': start,
                                'end': end,
                                'word': word,
                                'replacement': replacement
                            })
                    
                    all_entities.extend(additional_entities)
                    all_entities.sort(key=lambda x: x['start'], reverse=True)

                    # Create anonymized text
                    anonymized_text = input_text
                    for entity in all_entities:
                        replacement = entity.get('replacement', anonymizer.replacement_dict.get(entity['entity'], '[UNKNOWN]'))
                        anonymized_text = anonymized_text[:entity['start']] + replacement + anonymized_text[entity['end']:]

                    # Store results in session state
                    st.session_state.anonymized_text = anonymized_text
                    st.session_state.detected_entities = all_entities

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter some text to anonymize.")

    with col2:
        st.subheader("Results")
        
        # Display anonymized text
        if 'anonymized_text' in st.session_state:
            st.markdown("**Anonymized Text:**")
            st.text_area("", st.session_state.anonymized_text, height=300)
            
            # Display detected entities with case information
            st.markdown("**Detected Entities:**")
            for entity in sorted(st.session_state.detected_entities, key=lambda x: x['start']):
                case_info = ""
                if entity['entity'] == 'PER':
                    if entity['word'].isupper():
                        case_info = " (UPPERCASE)"
                    elif entity['word'].islower():
                        case_info = " (lowercase)"
                    elif entity['word'].istitle():
                        case_info = " (Title Case)"
                st.markdown(f"- **Text:** '{entity['word']}' → **Label:** {entity['entity']}{case_info} → **Replacement:** {entity.get('replacement', '[UNKNOWN]')}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Model: dbmdz/bert-large-cased-finetuned-conll03-english</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()