# 🤖 BERT-Based Data Anonymizer

A lightweight, scalable, and privacy-compliant **text anonymization tool** powered by BERT-based Named Entity Recognition (NER). This project is designed for developers, researchers, and enterprises seeking to anonymize sensitive information such as names, organizations, emails, phone numbers, and locations in text data—while preserving its analytical utility.

---

## 🚀 Features

- 🔍 **NER with BERT**: Uses `dbmdz/bert-large-cased-finetuned-conll03-english` via Hugging Face's API.
- 🧠 **Context-Aware Anonymization**: Detects and replaces entities like `PERSON`, `ORG`, `LOC`, and `MISC` with semantic understanding.
- 📧 **Custom Regex Detection**: Adds support for email and phone number anonymization.
- ⚙️ **Modular Design**: Easy to extend or integrate into larger data processing pipelines.
- ✅ **Privacy First**: Designed with GDPR, HIPAA, and CCPA compliance in mind.

---

## 🛠️ Installation

1. **Clone the repository**
git clone https://github.com/your-username/bert-data-anonymizer.git
cd bert-data-anonymizer

2. Install dependencies
pip install -r requirements.txt

3.Configure environment Create a .env file in the root directory and add your Hugging Face API token:
HF_TOKENS=your_huggingface_api_key
🔐 You can get your token from https://huggingface.co/settings/tokens



📦 Usage
python data_anonymizer.py
Or use the module programmatically:
from data_anonymizer import DataAnonymizer
anonymizer = DataAnonymizer()
text = "John Smith works at Google. Email: john@google.com"
anonymized = anonymizer.anonymize_text(text)
print(anonymized)



🧪 Example Input/Output
Input:

John Smith works at Microsoft in New York City.
You can email him at john.smith@microsoft.com or call +1-555-123-4567.
Output:

[Person] [SURNAME] works at [ORGANIZATION] in [LOCATION].
You can email him at [EMAIL] or call [PHONE].



🧱 Project Structure

├── data_anonymizer.py   # Main anonymization logic
├── .env                 # Hugging Face API Key (not committed)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation



🔍 System Requirements
Python 3.8+

Internet connection (for Hugging Face API calls)

No GPU required



📌 Roadmap
 Add support for offline inference using Hugging Face Transformers

 Extend NER coverage with domain-specific models (e.g., medical, legal)

 Develop a Streamlit-based web UI

 Add support for multiple languages
