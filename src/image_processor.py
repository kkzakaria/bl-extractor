import logging
import cv2
import numpy as np
from PIL import Image
from typing import Optional
from .paddleocr_processor import PaddleOCRProcessor

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Processeur pour l'extraction de texte depuis des images"""
    
    def __init__(self):
        self.confidence_threshold = 0.5
        self.paddleocr_processor = PaddleOCRProcessor()
    
    async def extract_text(self, image_path: str, ocr_method: str = "paddleocr") -> str:
        """
        Extrait le texte d'une image
        
        Args:
            image_path: Chemin vers le fichier image
            ocr_method: Méthode OCR à utiliser
            
        Returns:
            str: Texte extrait
        """
        try:
            # Utiliser PaddleOCR en priorité
            if ocr_method == "paddleocr" and self.paddleocr_processor.is_available():
                extracted_text = await self.paddleocr_processor.extract_text(image_path)
                logger.info("✅ PaddleOCR utilisé pour l'extraction")
                return extracted_text
            
            # Fallback vers Tesseract si demandé ou si PaddleOCR indisponible
            elif ocr_method == "tesseract" or not self.paddleocr_processor.is_available():
                # Préprocesser l'image pour Tesseract
                processed_image = await self._preprocess_image(image_path)
                extracted_text = await self._ocr_with_tesseract_fallback(processed_image)
                logger.info("✅ Tesseract fallback utilisé")
                return extracted_text
            
            # Fallback intelligent
            else:
                if self.paddleocr_processor.is_available():
                    return await self.paddleocr_processor.extract_text(image_path)
                else:
                    processed_image = await self._preprocess_image(image_path)
                    return await self._ocr_with_tesseract_fallback(processed_image)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction image: {str(e)}")
            raise Exception(f"Erreur lors de l'extraction image: {str(e)}")
    
    async def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Préprocesse l'image pour améliorer la qualité OCR
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            np.ndarray: Image préprocessée
        """
        try:
            # Charger l'image
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Augmenter la résolution si l'image est trop petite
            height, width = gray.shape
            if height < 1000 or width < 1000:
                scale_factor = max(1000 / height, 1000 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Appliquer un filtre pour réduire le bruit
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Améliorer le contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Binarisation adaptative
            binary = cv2.adaptiveThreshold(
                enhanced, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 
                2
            )
            
            # Fermeture morphologique pour connecter les caractères
            kernel = np.ones((1,1), np.uint8)
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            logger.error(f"Erreur préprocessing image: {str(e)}")
            # Retourner l'image originale en cas d'erreur
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    async def _ocr_with_tesseract_fallback(self, image: np.ndarray) -> str:
        """
        Applique l'OCR avec Tesseract en fallback
        
        Args:
            image: Image préprocessée
            
        Returns:
            str: Texte extrait
        """
        try:
            import pytesseract
            
            # Convertir en PIL Image
            pil_image = Image.fromarray(image)
            
            # Configuration Tesseract optimisée pour les documents
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,:-/()'
            
            # Extraire le texte
            text = pytesseract.image_to_string(
                pil_image, 
                lang='eng+fra',  # Anglais + Français
                config=custom_config
            )
            
            return text.strip()
            
        except ImportError:
            logger.warning("❌ Tesseract non disponible")
            return ""
        except Exception as e:
            logger.error(f"❌ Erreur Tesseract: {str(e)}")
            return ""
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore la qualité de l'image pour l'OCR
        
        Args:
            image: Image en niveaux de gris
            
        Returns:
            np.ndarray: Image améliorée
        """
        try:
            # Correction gamma
            gamma = 1.2
            lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(image, lookup_table)
            
            # Netteté
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gamma_corrected, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Erreur amélioration qualité: {str(e)}")
            return image