import logging
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import os

from .models import BillOfLadingData
from .docling_processor import DoclingProcessor
from .llm_enhancer import LLMEnhancer
from .pdf_processor import PDFProcessor
from .image_processor import ImageProcessor
from .text_parser import TextParser

logger = logging.getLogger(__name__)

class AdvancedBLExtractor:
    """Extracteur avancé avec Docling + LLM + fallbacks"""
    
    def __init__(self):
        self.docling_processor = DoclingProcessor()
        self.llm_enhancer = LLMEnhancer()
        self.pdf_processor = PDFProcessor()
        self.image_processor = ImageProcessor()
        self.text_parser = TextParser()
    
    async def extract(
        self,
        file_content: bytes,
        filename: str,
        ocr_method: str = "paddleocr",
        use_llm: bool = True,
        use_docling: bool = True
    ) -> BillOfLadingData:
        """
        Extraction avancée avec stratégie multi-niveaux
        
        Stratégie:
        1. Docling (extraction structurée) - PDF uniquement
        2. LLM + données structurées (amélioration)
        3. Fallback OCR + LLM
        4. Fallback OCR + Regex
        """
        
        file_extension = Path(filename).suffix.lower()
        
        # Créer fichier temporaire
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            logger.info(f"Extraction avancée démarrée pour {filename}")
            
            # Étape 1: Extraction structurée avec Docling (PDF uniquement)
            structured_data = None
            if file_extension == '.pdf' and use_docling and self.docling_processor.is_available():
                try:
                    structured_data = await self.docling_processor.extract_structured_data(tmp_file_path)
                    logger.info("✅ Docling: Données structurées extraites")
                    
                    # Si Docling réussit complètement, essayer l'extraction directe
                    if self._is_structured_data_complete(structured_data):
                        docling_text = await self.docling_processor.extract_text(tmp_file_path)
                        
                        if use_llm and self.llm_enhancer.is_available():
                            # LLM avec données structurées
                            llm_result = await self.llm_enhancer.enhance_extraction(docling_text, structured_data)
                            if llm_result and llm_result.extraction_confidence > 0.8:
                                llm_result.extraction_method = "docling_llm_gemma3"
                                logger.info(f"✅ Docling + LLM: Confiance {llm_result.extraction_confidence:.2f}")
                                return llm_result
                        
                except Exception as e:
                    logger.warning(f"⚠️ Docling échoué: {str(e)}")
            
            # Étape 2: Extraction OCR classique
            if file_extension == '.pdf':
                extracted_text = await self.pdf_processor.extract_text(tmp_file_path, ocr_method)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                extracted_text = await self.image_processor.extract_text(tmp_file_path, ocr_method)
            else:
                raise ValueError(f"Type de fichier non supporté: {file_extension}")
            
            logger.info(f"📄 OCR: {len(extracted_text)} caractères extraits")
            
            # Étape 3: LLM avec OCR (+ données structurées si disponibles)
            if use_llm and self.llm_enhancer.is_available():
                llm_result = await self.llm_enhancer.enhance_extraction(extracted_text, structured_data)
                if llm_result and llm_result.extraction_confidence > 0.5:
                    method = f"ocr_llm_gemma3"
                    if structured_data:
                        method = f"docling_ocr_llm_gemma3"
                    llm_result.extraction_method = method
                    logger.info(f"✅ OCR + LLM: Confiance {llm_result.extraction_confidence:.2f}")
                    return llm_result
            
            # Étape 4: Fallback regex
            logger.warning("🔄 Fallback vers extraction regex")
            regex_result = await self.text_parser.parse(extracted_text)
            
            method = f"{ocr_method}_regex"
            if structured_data:
                method = f"docling_{ocr_method}_regex"
            
            regex_result.extraction_method = method
            regex_result.raw_text = extracted_text
            
            logger.info(f"📊 Regex: Confiance {regex_result.extraction_confidence:.2f}")
            return regex_result
            
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def _is_structured_data_complete(self, structured_data: Dict[str, Any]) -> bool:
        """Vérifie si les données structurées sont suffisamment complètes"""
        if not structured_data:
            return False
        
        bl_sections = structured_data.get("bill_of_lading_sections", {})
        if not bl_sections:
            return False
        
        # Vérifier qu'on a au moins les sections principales
        required_sections = ["header_info", "parties", "ports"]
        found_sections = sum(1 for section in required_sections if bl_sections.get(section))
        
        return found_sections >= 2  # Au moins 2 sections sur 3
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Retourne les capacités disponibles avec informations GPU"""
        # Récupérer les informations GPU depuis PaddleOCR
        paddleocr_gpu_info = {}
        if hasattr(self.pdf_processor, 'paddleocr_processor') and self.pdf_processor.paddleocr_processor:
            paddleocr_gpu_info = self.pdf_processor.paddleocr_processor.get_gpu_info()
        elif hasattr(self.image_processor, 'paddleocr_processor') and self.image_processor.paddleocr_processor:
            paddleocr_gpu_info = self.image_processor.paddleocr_processor.get_gpu_info()
        
        base_capabilities = {
            "docling_available": self.docling_processor.is_available(),
            "llm_available": self.llm_enhancer.is_available(),
            "pdf_support": True,
            "image_support": True,
            "structured_extraction": self.docling_processor.is_available(),
            "intelligent_parsing": self.llm_enhancer.is_available()
        }
        
        # Ajouter les informations GPU si disponibles
        if paddleocr_gpu_info:
            gpu_detector = paddleocr_gpu_info.get("gpu_detector", {})
            performance_estimate = paddleocr_gpu_info.get("performance_estimate", {})
            
            base_capabilities.update({
                "gpu_acceleration": gpu_detector.get("recommended_use_gpu", False),
                "nvidia_gpu": gpu_detector.get("nvidia_gpu", False),
                "cuda_available": gpu_detector.get("cuda_available", False),
                "gpu_memory_mb": gpu_detector.get("gpu_memory", 0),
                "gpu_count": gpu_detector.get("gpu_count", 0),
                "expected_speedup": performance_estimate.get("expected_speedup", 1.0),
                "current_ocr_device": paddleocr_gpu_info.get("current_device", "CPU")
            })
        else:
            base_capabilities.update({
                "gpu_acceleration": False,
                "nvidia_gpu": False,
                "cuda_available": False,
                "gpu_memory_mb": 0,
                "gpu_count": 0,
                "expected_speedup": 1.0,
                "current_ocr_device": "CPU"
            })
        
        return base_capabilities
    
    def get_recommended_strategy(self, file_extension: str) -> str:
        """Retourne la stratégie recommandée selon le type de fichier"""
        capabilities = self.get_capabilities()
        
        if file_extension.lower() == '.pdf':
            if capabilities["docling_available"] and capabilities["llm_available"]:
                return "docling_llm_optimal"
            elif capabilities["llm_available"]:
                return "ocr_llm_standard"
            else:
                return "ocr_regex_basic"
        else:  # Images
            if capabilities["llm_available"]:
                return "ocr_llm_standard"
            else:
                return "ocr_regex_basic"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de performance détaillées"""
        stats = {}
        
        # Statistiques PaddleOCR
        if hasattr(self.pdf_processor, 'paddleocr_processor') and self.pdf_processor.paddleocr_processor:
            stats["paddleocr_pdf"] = self.pdf_processor.paddleocr_processor.get_performance_summary()
        
        if hasattr(self.image_processor, 'paddleocr_processor') and self.image_processor.paddleocr_processor:
            stats["paddleocr_image"] = self.image_processor.paddleocr_processor.get_performance_summary()
        
        # Capacités globales
        stats["capabilities"] = self.get_capabilities()
        
        return stats
    
    async def warmup_system(self):
        """Réchauffe tous les composants pour des performances optimales"""
        logger.info("🔥 Réchauffage du système en cours...")
        
        # Réchauffer PaddleOCR GPU si disponible
        if hasattr(self.pdf_processor, 'paddleocr_processor') and self.pdf_processor.paddleocr_processor:
            await self.pdf_processor.paddleocr_processor.warmup_gpu()
        
        if hasattr(self.image_processor, 'paddleocr_processor') and self.image_processor.paddleocr_processor:
            await self.image_processor.paddleocr_processor.warmup_gpu()
        
        logger.info("✅ Système réchauffé et prêt")