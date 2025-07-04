import io
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import os

from .models import BillOfLadingData
from .pdf_processor import PDFProcessor
from .image_processor import ImageProcessor
from .text_parser import TextParser
from .llm_enhancer import LLMEnhancer

logger = logging.getLogger(__name__)

class BLExtractor:
    """Classe principale pour l'extraction de données de connaissements"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.image_processor = ImageProcessor()
        self.text_parser = TextParser()
        self.llm_enhancer = LLMEnhancer()
    
    async def extract(
        self, 
        file_content: bytes, 
        filename: str, 
        ocr_method: str = "tesseract",
        use_llm: bool = True
    ) -> BillOfLadingData:
        """
        Extrait les données d'un connaissement depuis un fichier
        
        Args:
            file_content: Contenu du fichier en bytes
            filename: Nom du fichier
            ocr_method: Méthode OCR à utiliser ("tesseract", "easyocr")
            use_llm: Utiliser le LLM pour améliorer l'extraction
            
        Returns:
            BillOfLadingData: Données extraites structurées
        """
        try:
            # Déterminer le type de fichier
            file_extension = Path(filename).suffix.lower()
            
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(
                suffix=file_extension, 
                delete=False
            ) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Extraire le texte selon le type de fichier
                if file_extension == '.pdf':
                    extracted_text = await self.pdf_processor.extract_text(
                        tmp_file_path, ocr_method
                    )
                elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                    extracted_text = await self.image_processor.extract_text(
                        tmp_file_path, ocr_method
                    )
                else:
                    raise ValueError(f"Type de fichier non supporté: {file_extension}")
                
                # Stratégie d'extraction hybride avancée
                structured_data = None
                
                # Essayer d'obtenir des données structurées avec Docling (pour PDF)
                if file_extension == '.pdf' and hasattr(self.pdf_processor, 'docling_processor'):
                    try:
                        if self.pdf_processor.docling_processor.is_available():
                            structured_data = await self.pdf_processor.docling_processor.extract_structured_data(tmp_file_path)
                            logger.info("Données structurées Docling obtenues")
                    except Exception as e:
                        logger.warning(f"Docling structured extraction échoué: {str(e)}")
                
                if use_llm and self.llm_enhancer.is_available():
                    # Essayer d'abord avec le LLM (avec données structurées si disponibles)
                    llm_data = await self.llm_enhancer.enhance_extraction(extracted_text, structured_data)
                    
                    if llm_data and llm_data.extraction_confidence > 0.5:
                        logger.info(f"Extraction LLM réussie avec confiance: {llm_data.extraction_confidence}")
                        return llm_data
                    else:
                        logger.warning("Extraction LLM échouée, fallback vers regex")
                
                # Fallback vers le parser regex
                parsed_data = await self.text_parser.parse(extracted_text)
                
                # Ajouter les métadonnées d'extraction
                extraction_method = f"{ocr_method}_regex"
                if structured_data:
                    extraction_method = f"docling_{ocr_method}_regex"
                
                parsed_data.extraction_method = extraction_method
                parsed_data.raw_text = extracted_text
                
                return parsed_data
                
            finally:
                # Nettoyer le fichier temporaire
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction: {str(e)}")
            raise Exception(f"Erreur lors de l'extraction: {str(e)}")
    
    def get_supported_formats(self) -> Dict[str, list]:
        """Retourne les formats supportés"""
        return {
            "pdf": [".pdf"],
            "images": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
        }