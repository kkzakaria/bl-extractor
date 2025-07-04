from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
from typing import Optional

from src.advanced_extractor import AdvancedBLExtractor
from src.models import BillOfLadingData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bill of Lading Extractor",
    description="Service d'extraction de données de connaissements (PDF/Images) vers JSON",
    version="1.0.0"
)

extractor = AdvancedBLExtractor()

@app.get("/")
async def root():
    return {"message": "Bill of Lading Extractor API"}

@app.post("/extract", response_model=BillOfLadingData)
async def extract_bill_of_lading(
    file: UploadFile = File(...),
    ocr_method: Optional[str] = "paddleocr",
    use_llm: Optional[bool] = True,
    use_docling: Optional[bool] = True
):
    """
    Extrait les données d'un connaissement depuis un PDF ou une image
    
    Args:
        file: Fichier PDF ou image
        ocr_method: Méthode OCR ("paddleocr", "tesseract")
        use_llm: Utiliser le LLM Gemma3:12b pour améliorer l'extraction
        use_docling: Utiliser Docling pour l'extraction structurée (PDF uniquement)
    """
    try:
        # Vérifier le type de fichier
        if not file.content_type.startswith(('image/', 'application/pdf')):
            raise HTTPException(
                status_code=400, 
                detail="Seuls les fichiers PDF et images sont acceptés"
            )
        
        # Lire le contenu du fichier
        file_content = await file.read()
        
        # Extraire les données
        extracted_data = await extractor.extract(
            file_content=file_content,
            filename=file.filename,
            ocr_method=ocr_method,
            use_llm=use_llm,
            use_docling=use_docling
        )
        
        return extracted_data
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    capabilities = extractor.get_capabilities()
    return {
        "status": "healthy", 
        "service": "bl-extractor",
        "capabilities": capabilities,
        "version": "2.0.0-docling"
    }

@app.get("/capabilities")
async def get_capabilities():
    """Retourne les capacités du service"""
    capabilities = extractor.get_capabilities()
    
    recommendations = {
        "pdf": extractor.get_recommended_strategy(".pdf"),
        "image": extractor.get_recommended_strategy(".jpg")
    }
    
    return {
        "capabilities": capabilities,
        "recommended_strategies": recommendations,
        "extraction_methods": [
            "docling_llm_gemma3",        # Optimal pour PDF
            "docling_paddleocr_llm_gemma3", # Hybride PDF
            "paddleocr_llm_gemma3",      # Standard images
            "paddleocr_regex",           # Fallback moderne
            "tesseract_regex",           # Fallback legacy
            "docling_paddleocr_regex"    # Fallback avec structure
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)