import logging
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
import tempfile
import os
import time

try:
    from .gpu_detector import GPUDetector
except ImportError:
    from gpu_detector import GPUDetector

logger = logging.getLogger(__name__)

class PaddleOCRProcessor:
    """Processeur PaddleOCR pour l'extraction de texte depuis des images et PDF avec support GPU"""
    
    def __init__(self):
        self.ocr_engine = None
        self.gpu_detector = GPUDetector()
        self.available = self._init_paddle_ocr()
        self.confidence_threshold = 0.5
        self.use_gpu = self.gpu_detector.should_use_gpu()
        self.performance_stats = {"total_extractions": 0, "total_time": 0.0, "avg_time": 0.0}
    
    def _init_paddle_ocr(self) -> bool:
        """Initialise PaddleOCR avec détection automatique GPU"""
        try:
            # Configurer la suppression des logs PaddleOCR avant l'import
            os.environ['DISABLE_AUTO_LOGGING_CONFIG'] = '1'
            
            from paddleocr import PaddleOCR, logger as paddleocr_logger
            
            # Configurer le logger PaddleOCR selon la doc officielle
            paddleocr_logger.setLevel(logging.WARNING)
            
            # Afficher le statut GPU
            self.gpu_detector.log_gpu_status()
            
            # Déterminer si utiliser GPU
            use_gpu = self.gpu_detector.should_use_gpu()
            
            # Initialiser PaddleOCR avec configuration optimale pour v3.x
            ocr_config = {
                'use_angle_cls': True,  # Classification d'angle pour rotation
                'lang': 'en',           # Anglais par défaut
                'device': 'gpu:0' if use_gpu else 'cpu',  # Device pour PaddleOCR 3.x
                'enable_mkldnn': True,  # Optimisations CPU
            }
            
            # Ajouter les paramètres CPU seulement si on n'utilise pas GPU
            if not use_gpu:
                ocr_config['cpu_threads'] = 4
            
            self.ocr_engine = PaddleOCR(**ocr_config)
            
            device_type = "GPU" if use_gpu else "CPU"
            logger.info(f"✅ PaddleOCR initialisé avec succès ({device_type})")
            
            # Afficher les estimations de performance
            perf_info = self.gpu_detector.get_performance_estimate()
            if perf_info["gpu_acceleration"]:
                logger.info(f"🚀 {perf_info['recommendation']}")
            
            return True
            
        except ImportError:
            logger.warning("❌ PaddleOCR non disponible. Installation: pip install paddlepaddle paddleocr")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur initialisation PaddleOCR: {str(e)}")
            # Fallback vers CPU si GPU échoue
            if self.gpu_detector.should_use_gpu():
                logger.info("🔄 Tentative de fallback vers CPU...")
                try:
                    self.ocr_engine = PaddleOCR(
                        use_angle_cls=True,
                        lang='en',
                        device='cpu'
                    )
                    logger.info("✅ PaddleOCR initialisé en mode CPU (fallback)")
                    return True
                except Exception as e2:
                    logger.error(f"❌ Fallback CPU échoué: {str(e2)}")
            return False
    
    async def extract_text(self, image_path: str, language: str = "en") -> str:
        """
        Extrait le texte d'une image avec PaddleOCR (GPU optimisé)
        
        Args:
            image_path: Chemin vers le fichier image
            language: Langue pour l'OCR ("en", "fr", "ch", etc.)
            
        Returns:
            str: Texte extrait
        """
        if not self.available:
            raise Exception("PaddleOCR non disponible")
        
        start_time = time.time()
        
        try:
            device_info = "GPU" if self.use_gpu else "CPU"
            logger.info(f"Extraction PaddleOCR démarrée ({device_info}): {image_path}")
            
            # Préprocesser l'image si nécessaire
            processed_image_path = await self._preprocess_image(image_path)
            
            # Exécuter PaddleOCR avec mesure de performance
            ocr_start = time.time()
            results = self.ocr_engine.ocr(processed_image_path, cls=True)
            ocr_time = time.time() - ocr_start
            
            # Extraire le texte des résultats
            extracted_text = self._extract_text_from_results(results)
            
            # Nettoyer le fichier temporaire si créé
            if processed_image_path != image_path and os.path.exists(processed_image_path):
                os.unlink(processed_image_path)
            
            # Mettre à jour les statistiques de performance
            total_time = time.time() - start_time
            self._update_performance_stats(total_time)
            
            logger.info(f"✅ PaddleOCR ({device_info}): {len(extracted_text)} caractères extraits en {ocr_time:.2f}s")
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
            "de",    # Allemand
            "ko",    # Coréen
            "ja"     # Japonais
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
    
    def _update_performance_stats(self, extraction_time: float):
        """Met à jour les statistiques de performance"""
        self.performance_stats["total_extractions"] += 1
        self.performance_stats["total_time"] += extraction_time
        self.performance_stats["avg_time"] = (
            self.performance_stats["total_time"] / self.performance_stats["total_extractions"]
        )
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Retourne les informations GPU détaillées"""
        return {
            "gpu_detector": self.gpu_detector.get_gpu_info(),
            "performance_estimate": self.gpu_detector.get_performance_estimate(),
            "current_device": "GPU" if self.use_gpu else "CPU",
            "performance_stats": self.performance_stats.copy()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances"""
        stats = self.performance_stats.copy()
        gpu_info = self.gpu_detector.get_gpu_info()
        
        return {
            "device": "GPU" if self.use_gpu else "CPU",
            "gpu_available": gpu_info.get("nvidia_gpu", False),
            "gpu_memory_mb": gpu_info.get("gpu_memory", 0),
            "total_extractions": stats["total_extractions"],
            "avg_extraction_time": round(stats["avg_time"], 3),
            "total_processing_time": round(stats["total_time"], 2),
            "estimated_speedup": self.gpu_detector.get_performance_estimate().get("expected_speedup", 1.0)
        }
    
    async def warmup_gpu(self):
        """Réchauffe le GPU pour des performances optimales"""
        if not self.use_gpu or not self.available:
            return
        
        try:
            logger.info("🔥 Réchauffage GPU en cours...")
            
            # Créer une image de test minimale
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, test_image)
                
                # Exécuter une extraction de test
                start_time = time.time()
                _ = self.ocr_engine.ocr(tmp_file.name, cls=False)
                warmup_time = time.time() - start_time
                
                # Nettoyer
                os.unlink(tmp_file.name)
                
                logger.info(f"✅ GPU réchauffé en {warmup_time:.2f}s")
                
        except Exception as e:
            logger.warning(f"⚠️ Échec réchauffage GPU: {str(e)}")
    
    def reset_performance_stats(self):
        """Remet à zéro les statistiques de performance"""
        self.performance_stats = {"total_extractions": 0, "total_time": 0.0, "avg_time": 0.0}
        logger.info("📊 Statistiques de performance réinitialisées")