import logging
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
import tempfile
import os

logger = logging.getLogger(__name__)

class PaddleOCRProcessor:
    """Processeur PaddleOCR pour l'extraction de texte depuis des images et PDF"""
    
    def __init__(self):
        self.ocr_engine = None
        self.available = self._init_paddle_ocr()
        self.confidence_threshold = 0.5
    
    def _init_paddle_ocr(self) -> bool:
        """Initialise PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            
            # Initialiser PaddleOCR avec les langues supportées
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,  # Classification d'angle pour rotation
                lang='en',           # Anglais par défaut
                use_gpu=False,       # CPU par défaut (changeable)
                show_log=False       # Réduire les logs
            )
            
            logger.info("✅ PaddleOCR initialisé avec succès")
            return True
            
        except ImportError:
            logger.warning("❌ PaddleOCR non disponible. Installation: pip install paddlepaddle paddleocr")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur initialisation PaddleOCR: {str(e)}")
            return False
    
    async def extract_text(self, image_path: str, language: str = "en") -> str:
        """
        Extrait le texte d'une image avec PaddleOCR
        
        Args:
            image_path: Chemin vers le fichier image
            language: Langue pour l'OCR ("en", "fr", "ch", etc.)
            
        Returns:
            str: Texte extrait
        """
        if not self.available:
            raise Exception("PaddleOCR non disponible")
        
        try:
            logger.info(f"Extraction PaddleOCR démarrée: {image_path}")
            
            # Préprocesser l'image si nécessaire
            processed_image_path = await self._preprocess_image(image_path)
            
            # Exécuter PaddleOCR
            results = self.ocr_engine.ocr(processed_image_path, cls=True)
            
            # Extraire le texte des résultats
            extracted_text = self._extract_text_from_results(results)
            
            # Nettoyer le fichier temporaire si créé
            if processed_image_path != image_path and os.path.exists(processed_image_path):
                os.unlink(processed_image_path)
            
            logger.info(f"✅ PaddleOCR: {len(extracted_text)} caractères extraits")
            return extracted_text
            
        except Exception as e:
            logger.error(f"❌ Erreur PaddleOCR: {str(e)}")
            raise Exception(f"Erreur PaddleOCR: {str(e)}")
    
    async def extract_structured_text(self, image_path: str, language: str = "en") -> Dict[str, Any]:
        """
        Extrait le texte avec informations de structure (positions, confiance)
        
        Args:
            image_path: Chemin vers le fichier image
            language: Langue pour l'OCR
            
        Returns:
            Dict: Texte structuré avec métadonnées
        """
        if not self.available:
            raise Exception("PaddleOCR non disponible")
        
        try:
            processed_image_path = await self._preprocess_image(image_path)
            results = self.ocr_engine.ocr(processed_image_path, cls=True)
            
            structured_data = {
                "text": self._extract_text_from_results(results),
                "text_blocks": [],
                "confidence_scores": [],
                "bounding_boxes": []
            }
            
            # Extraire les informations détaillées
            if results and results[0]:
                for line_result in results[0]:
                    if line_result and len(line_result) >= 2:
                        bbox = line_result[0]  # Coordonnées
                        text_info = line_result[1]  # (texte, confiance)
                        
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            
                            structured_data["text_blocks"].append(text)
                            structured_data["confidence_scores"].append(confidence)
                            structured_data["bounding_boxes"].append(bbox)
            
            # Nettoyer
            if processed_image_path != image_path and os.path.exists(processed_image_path):
                os.unlink(processed_image_path)
            
            return structured_data
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction structurée PaddleOCR: {str(e)}")
            raise Exception(f"Erreur extraction structurée PaddleOCR: {str(e)}")
    
    async def _preprocess_image(self, image_path: str) -> str:
        """
        Préprocesse l'image pour améliorer la qualité OCR
        
        Args:
            image_path: Chemin vers l'image originale
            
        Returns:
            str: Chemin vers l'image préprocessée
        """
        try:
            # Charger l'image
            image = cv2.imread(image_path)
            
            if image is None:
                logger.warning("Impossible de charger l'image pour preprocessing")
                return image_path
            
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Améliorer la qualité
            processed = self._enhance_image_quality(gray)
            
            # Sauvegarder l'image préprocessée
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, processed)
                return tmp_file.name
            
        except Exception as e:
            logger.warning(f"Preprocessing échoué: {str(e)}, utilisation image originale")
            return image_path
    
    def _enhance_image_quality(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Améliore la qualité de l'image pour PaddleOCR
        
        Args:
            gray_image: Image en niveaux de gris
            
        Returns:
            np.ndarray: Image améliorée
        """
        try:
            # Redimensionner si trop petite
            height, width = gray_image.shape
            if height < 800 or width < 800:
                scale_factor = max(800 / height, 800 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray_image = cv2.resize(gray_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Débruitage
            denoised = cv2.fastNlMeansDenoising(gray_image)
            
            # Améliorer le contraste avec CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Amélioration qualité échouée: {str(e)}")
            return gray_image
    
    def _extract_text_from_results(self, results: List) -> str:
        """
        Extrait le texte des résultats PaddleOCR
        
        Args:
            results: Résultats bruts de PaddleOCR
            
        Returns:
            str: Texte extrait et nettoyé
        """
        text_lines = []
        
        try:
            if results and results[0]:
                for line_result in results[0]:
                    if line_result and len(line_result) >= 2:
                        text_info = line_result[1]
                        
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            
                            # Filtrer par confiance
                            if confidence >= self.confidence_threshold:
                                text_lines.append(text)
                        elif isinstance(text_info, str):
                            text_lines.append(text_info)
            
            # Joindre les lignes avec des espaces
            return ' '.join(text_lines).strip()
            
        except Exception as e:
            logger.error(f"Erreur extraction texte PaddleOCR: {str(e)}")
            return ""
    
    def is_available(self) -> bool:
        """Retourne True si PaddleOCR est disponible"""
        return self.available
    
    def get_supported_languages(self) -> List[str]:
        """Retourne la liste des langues supportées"""
        return [
            "en",    # Anglais
            "fr",    # Français
            "ch",    # Chinois
            "german", # Allemand
            "korean", # Coréen
            "japan"   # Japonais
        ]
    
    def set_confidence_threshold(self, threshold: float):
        """Configure le seuil de confiance minimum"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Seuil de confiance PaddleOCR: {self.confidence_threshold}")
    
    async def benchmark_vs_tesseract(self, image_path: str) -> Dict[str, Any]:
        """
        Compare PaddleOCR avec Tesseract sur une image
        
        Args:
            image_path: Chemin vers l'image de test
            
        Returns:
            Dict: Résultats comparatifs
        """
        results = {
            "paddleocr": {"available": self.available, "text": "", "word_count": 0, "time": 0},
            "tesseract": {"available": False, "text": "", "word_count": 0, "time": 0}
        }
        
        # Test PaddleOCR
        if self.available:
            import time
            start_time = time.time()
            try:
                paddle_text = await self.extract_text(image_path)
                results["paddleocr"]["text"] = paddle_text
                results["paddleocr"]["word_count"] = len(paddle_text.split())
                results["paddleocr"]["time"] = time.time() - start_time
            except Exception as e:
                results["paddleocr"]["error"] = str(e)
        
        # Test Tesseract (si disponible)
        try:
            import pytesseract
            start_time = time.time()
            tesseract_text = pytesseract.image_to_string(image_path, lang='eng')
            results["tesseract"]["available"] = True
            results["tesseract"]["text"] = tesseract_text.strip()
            results["tesseract"]["word_count"] = len(tesseract_text.split())
            results["tesseract"]["time"] = time.time() - start_time
        except ImportError:
            results["tesseract"]["error"] = "Tesseract non disponible"
        except Exception as e:
            results["tesseract"]["error"] = str(e)
        
        return results