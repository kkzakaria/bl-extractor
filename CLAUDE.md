# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Bill of Lading Extractor is a FastAPI-based service that extracts structured data from PDF and image files containing shipping documents (bills of lading). It uses a sophisticated multi-level extraction strategy combining modern OCR, document layout analysis, and LLM enhancement.

## Common Commands

### Development
```bash
# Start the API server
python main.py

# Run all tests
python -m pytest tests/

# Test specific integrations
python test_paddleocr_integration.py
python test_docling_integration.py
python test_llm_integration.py
python test_simple_llm.py
```

### Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies for PaddleOCR
pip install paddlepaddle paddleocr

# Install Tesseract (fallback OCR)
sudo apt-get install tesseract-ocr tesseract-ocr-fra libgl1-mesa-glx libglib2.0-0

# Install and setup Ollama LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:12b

# Install Docling for structured extraction
pip install docling
```

### API Usage
```bash
# Health check and capabilities
curl http://localhost:8000/health
curl http://localhost:8000/capabilities

# Extract with optimal strategy (Docling + PaddleOCR + LLM)
curl -X POST "http://localhost:8000/extract" \
  -F "file=@document.pdf" \
  -F "ocr_method=paddleocr" \
  -F "use_docling=true" \
  -F "use_llm=true"
```

## Architecture

The extraction system uses a cascading fallback strategy:

1. **Docling** (PDF only): Layout-aware structured extraction
2. **LLM Enhancement**: Gemma3:12b model via Ollama for intelligent parsing
3. **OCR Processing**: PaddleOCR (primary) or Tesseract (fallback)
4. **Regex Parsing**: Final fallback using pattern matching

### Core Components

- **`src/advanced_extractor.py`**: Main orchestrator implementing multi-level extraction strategy
- **`src/models.py`**: Pydantic models defining the structured output schema
- **`src/docling_processor.py`**: Handles PDF layout analysis and structured data extraction
- **`src/llm_enhancer.py`**: Interfaces with Ollama/Gemma3 for intelligent text processing
- **`src/pdf_processor.py`**: PDF text extraction with OCR fallback
- **`src/image_processor.py`**: Image preprocessing and OCR
- **`src/text_parser.py`**: Regex-based fallback parsing
- **`src/paddleocr_processor.py`**: Modern OCR implementation

### Extraction Methods

The system automatically selects the best available method:
- `docling_llm_gemma3`: Optimal for PDFs (Docling + LLM)
- `docling_paddleocr_llm_gemma3`: Hybrid PDF approach (Docling + PaddleOCR + LLM)
- `paddleocr_llm_gemma3`: Standard for images (PaddleOCR + LLM)
- `paddleocr_regex`: Modern fallback without LLM
- `tesseract_regex`: Legacy fallback
- `docling_paddleocr_regex`: Structured fallback without LLM

## Data Flow

1. **File Upload**: PDF or image via FastAPI endpoint
2. **Strategy Selection**: Based on file type and available capabilities
3. **Structured Extraction**: Docling extracts layout-aware data (PDF only)
4. **OCR Processing**: Extract text using PaddleOCR or Tesseract
5. **LLM Enhancement**: Gemma3 processes and structures the extracted text
6. **Fallback Chain**: If any step fails, cascade to next available method
7. **JSON Response**: Return structured BillOfLadingData model

## Key Dependencies

- **FastAPI**: Web framework and API
- **Docling**: PDF layout analysis and structured extraction
- **PaddleOCR**: Modern OCR engine (primary)
- **Tesseract**: OCR fallback
- **Ollama**: LLM inference server
- **Pydantic**: Data validation and serialization
- **OpenCV/Pillow**: Image processing

## Testing Strategy

The project includes integration tests for each major component:
- `test_paddleocr_integration.py`: OCR accuracy testing
- `test_docling_integration.py`: PDF structure extraction
- `test_llm_integration.py`: LLM enhancement validation
- `test_simple_llm.py`: Basic LLM functionality

## Configuration

The system automatically detects available capabilities and adjusts extraction strategy accordingly. No manual configuration required - the service gracefully degrades if optional dependencies (Docling, Ollama) are unavailable.

## Output Schema

The extracted data follows the `BillOfLadingData` model with nested structures for:
- Parties (shipper, consignee, notify party)
- Ports (loading, discharge, delivery)
- Transport details (vessel, voyage, dates)
- Cargo and container information
- Extraction metadata (confidence score, method used)