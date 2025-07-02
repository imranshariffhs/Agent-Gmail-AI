# Document Processing POC with AI

A proof-of-concept application for automated document processing using AI, specifically designed for handling business enquiries and quotations.

## 🌟 Features

- **Email Integration**: Automated Gmail attachment download and processing
- **Document Classification**: AI-powered classification of enquiries and quotations
- **Data Extraction**: Intelligent extraction of structured data from PDFs
- **Archive Management**: Organized storage and tracking of processed documents
- **Duplicate Detection**: Advanced file duplicate checking system

## 🏗️ Project Structure

```
Agent_AI/
├── classification_engine.py   # Document classification logic
├── download_email/           # Email attachment storage
├── extract_pdf_data.py       # PDF data extraction utilities
├── gmail_agent.py            # Gmail integration and management
├── image/                    # Processed document images
├── main.py                   # Application entry point
├── pdf_process.py           # PDF processing utilities
├── result_json/             # Extracted data storage
└── schemas/                 # JSON schemas for data validation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Gmail API credentials
- Google Cloud Vision API access
- Gemini API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Agent_AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## 📁 Archive Structure

The system maintains a hierarchical archive structure:

### Level 1: Document Categories
- `download_email/`: Raw email attachments
- `image/`: Processed document images
- `result_json/`: Extracted structured data

### Level 2: Processing Stages
- **Stage 1**: Raw document storage
- **Stage 2**: Image processing and text extraction
- **Stage 3**: Structured data extraction
- **Stage 4**: Validation and verification

### Level 3: Document Organization
- Timestamp-based naming convention
- Classification-based prefixes
- Original filename preservation
- Duplicate detection and handling

### Level 4: Metadata Management
- Document classification results
- Processing timestamps
- Extraction confidence scores
- Validation status

## 🔄 Processing Flow

1. **Email Download**
   - Automated attachment detection
   - Format validation
   - Duplicate checking

2. **Document Classification**
   - AI-powered content analysis
   - Section detection
   - Keyword matching
   - Bank details recognition

3. **Data Extraction**
   - Image preprocessing
   - Text extraction
   - Structure recognition
   - Data validation

4. **Archive Management**
   - Automated file organization
   - Metadata tracking
   - Processing status updates
   - Duplicate handling

## 📊 Logging and Tracking

The system maintains comprehensive logs:
- `email_download_log.xlsx`: Tracks all downloaded attachments
- Processing status and history
- Error tracking and handling
- Performance metrics

## 🛠️ Configuration

Key configuration options in `.env`:
```
GMAIL_CREDENTIALS_PATH=path/to/credentials.json
GOOGLE_CLOUD_VISION_KEY=your-vision-api-key
GEMINI_API_KEY=your-gemini-api-key
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 