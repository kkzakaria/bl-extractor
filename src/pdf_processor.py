import logging
import pdfplumber
import fitz  # PyMuPDF
from typing import Optional
from PIL import Image
import io
import tempfile
import os
from .docling_processor import DoclingProcessor
from .paddleocr_processor import PaddleOCRProcessor

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Processeur pour l'extraction de texte depuis des fichiers PDF"""
    
    def __init__(self):
        self.confidence_threshold = 0.5
        self.docling_processor = DoclingProcessor()
        self.paddleocr_processor = PaddleOCRProcessor()
    
    async def extract_text(self, pdf_path: str, ocr_method: str = "paddleocr") -> str:
        """
        Extrait le texte d'un PDF
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            ocr_method: Méthode OCR à utiliser
            
        Returns:
            str: Texte extrait
        """
        try:
            # 1. Essayer Docling d'abord (meilleure qualité pour documents structurés)
            if self.docling_processor.is_available():
                try:
                    docling_text = await self.docling_processor.extract_text(pdf_path)
                    if docling_text and len(docling_text.strip()) > 50:
                        logger.info("Extraction Docling réussie")
                        return docling_text
                except Exception as e:
                    logger.warning(f"Docling échoué, fallback: {str(e)}")
            
            # 2. Fallback: Essayer d'extraire le texte natif
            native_text = await self._extract_native_text(pdf_path)
            
            if native_text and len(native_text.strip()) > 50:
                logger.info("Texte natif extrait avec succès")
                return native_text
            
            # 3. Dernier recours: OCR
            logger.info(f"Texte natif insuffisant, utilisation de l'OCR: {ocr_method}")
            ocr_text = await self._extract_with_ocr(pdf_path, ocr_method)
            
            return ocr_text
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction PDF: {str(e)}")
            raise Exception(f"Erreur lors de l'extraction PDF: {str(e)}")
    
    async def _extract_native_text(self, pdf_path: str) -> str:
        """Extrait le texte natif du PDF"""
        try:
            text_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.warning(f"Erreur extraction texte natif: {str(e)}")
            return ""
    
    async def _extract_with_ocr(self, pdf_path: str, ocr_method: str) -> str:
        """Extrait le texte du PDF avec OCR"""
        try:
            text_content = []
            
            # Convertir les pages PDF en images
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Convertir la page en image avec haute résolution
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                
                # Sauvegarder temporairement l'image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    tmp_file.write(img_data)
                    tmp_image_path = tmp_file.name
                
                try:
                    # Appliquer l'OCR selon la méthode choisie
                    if ocr_method == "paddleocr" and self.paddleocr_processor.is_available():
                        page_text = await self.paddleocr_processor.extract_text(tmp_image_path)
                    elif ocr_method == "tesseract":
                        page_text = await self._ocr_with_tesseract_fallback(tmp_image_path)
                    else:
                        # Fallback intelligent
                        if self.paddleocr_processor.is_available():
                            page_text = await self.paddleocr_processor.extract_text(tmp_image_path)
                        else:
                            page_text = await self._ocr_with_tesseract_fallback(tmp_image_path)
                    
                    if page_text and page_text.strip():
                        text_content.append(page_text)
                        
                finally:
                    # Nettoyer le fichier temporaire
                    if os.path.exists(tmp_image_path):
                        os.unlink(tmp_image_path)
            
            pdf_document.close()
            
            if not text_content:
                logger.warning("Aucun texte extrait par OCR")
                return ""
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Erreur OCR PDF: {str(e)}")
            raise Exception(f"Erreur OCR PDF: {str(e)}")
    
    async def _ocr_with_tesseract_fallback(self, image_path: str) -> str:
        """Applique l'OCR avec Tesseract en fallback"""
        try:
            import pytesseract
            
            # Configuration Tesseract pour améliorer la précision
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,:-/()'
            
            text = pytesseract.image_to_string(
                image_path, 
                lang='eng+fra',  # Anglais + Français
                config=custom_config
            )
            
            logger.info("✅ Tesseract fallback utilisé")
            return text.strip()
            
        except ImportError:
            logger.warning("❌ Tesseract non disponible")
            return ""
        except Exception as e:
            logger.error(f"❌ Erreur Tesseract: {str(e)}")
            return ""