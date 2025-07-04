import logging
import os
import subprocess
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GPUDetector:
    """Détecteur de capacités GPU pour PaddleOCR"""
    
    def __init__(self):
        self._gpu_info = None
        self._paddle_gpu_available = None
        self._detect_gpu_capabilities()
    
    def _detect_gpu_capabilities(self):
        """Détecte les capacités GPU disponibles"""
        self._gpu_info = {
            "nvidia_gpu": self._check_nvidia_gpu(),
            "cuda_available": self._check_cuda(),
            "paddle_gpu_support": self._check_paddle_gpu(),
            "gpu_memory": self._get_gpu_memory(),
            "gpu_count": self._get_gpu_count(),
            "recommended_use_gpu": False
        }
        
        # Détection d'utilisation GPU effective (PyTorch pour Docling/EasyOCR)
        pytorch_gpu_available = self._check_pytorch_gpu()
        self._gpu_info["pytorch_gpu_support"] = pytorch_gpu_available
        
        # GPU utilisé si PyTorch peut l'utiliser (même si PaddleOCR ne peut pas)
        self._gpu_info["gpu_actually_used"] = (
            self._gpu_info["nvidia_gpu"] and 
            self._gpu_info["cuda_available"] and 
            pytorch_gpu_available
        )
        
        # Recommandation d'utilisation GPU pour PaddleOCR spécifiquement
        self._gpu_info["recommended_use_gpu"] = (
            self._gpu_info["nvidia_gpu"] and 
            self._gpu_info["cuda_available"] and 
            self._gpu_info["paddle_gpu_support"] and
            self._gpu_info["gpu_memory"] > 2000  # Au moins 2GB de VRAM
        )
        
        logger.info(f"GPU Detection: {self._gpu_info}")
    
    def _check_nvidia_gpu(self) -> bool:
        """Vérifie la présence d'une GPU NVIDIA"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def _check_cuda(self) -> bool:
        """Vérifie la disponibilité de CUDA"""
        try:
            # Vérifier CUDA via nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True
            
            # Vérifier via nvcc si disponible
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def _check_paddle_gpu(self) -> bool:
        """Vérifie si PaddlePaddle supporte GPU"""
        try:
            import paddle
            
            # Vérifier si PaddlePaddle a été compilé avec CUDA
            if hasattr(paddle, 'is_compiled_with_cuda'):
                cuda_compiled = paddle.is_compiled_with_cuda()
                if cuda_compiled:
                    # Vérifier si des GPU sont disponibles
                    gpu_count = paddle.device.cuda.device_count()
                    return gpu_count > 0
            
            return False
            
        except ImportError:
            logger.warning("PaddlePaddle non disponible pour vérification GPU")
            return False
        except Exception as e:
            logger.warning(f"Erreur vérification Paddle GPU: {str(e)}")
            return False
    
    def _check_pytorch_gpu(self) -> bool:
        """Vérifie si PyTorch supporte GPU (utilisé par Docling/EasyOCR)"""
        try:
            import torch
            # Vérifier si CUDA est disponible dans PyTorch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                return gpu_count > 0
            return False
        except ImportError:
            # PyTorch pas installé, probablement pas de GPU support
            return False
        except Exception as e:
            logger.warning(f"Erreur vérification PyTorch GPU: {str(e)}")
            return False
    
    def _get_gpu_memory(self) -> int:
        """Retourne la mémoire GPU disponible en MB"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                memory_lines = result.stdout.strip().split('\n')
                if memory_lines:
                    return int(memory_lines[0])
            return 0
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, Exception):
            return 0
    
    def _get_gpu_count(self) -> int:
        """Retourne le nombre de GPU disponibles"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_names = result.stdout.strip().split('\n')
                return len([name for name in gpu_names if name.strip()])
            return 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return 0
    
    def should_use_gpu(self) -> bool:
        """Retourne True si le GPU devrait être utilisé"""
        return self._gpu_info.get("recommended_use_gpu", False)
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Retourne les informations détaillées sur le GPU"""
        return self._gpu_info.copy()
    
    def get_paddle_device(self) -> str:
        """Retourne le device optimal pour PaddlePaddle"""
        if self.should_use_gpu():
            return "gpu"
        return "cpu"
    
    def get_performance_estimate(self) -> Dict[str, Any]:
        """Estime l'amélioration de performance attendue avec GPU"""
        if not self.should_use_gpu():
            return {
                "gpu_acceleration": False,
                "expected_speedup": 1.0,
                "recommendation": "GPU non disponible ou non recommandé"
            }
        
        # Estimation basée sur la mémoire GPU
        gpu_memory = self._gpu_info.get("gpu_memory", 0)
        if gpu_memory > 8000:  # 8GB+
            speedup = 5.0
        elif gpu_memory > 4000:  # 4GB+
            speedup = 3.0
        elif gpu_memory > 2000:  # 2GB+
            speedup = 2.0
        else:
            speedup = 1.5
        
        return {
            "gpu_acceleration": True,
            "expected_speedup": speedup,
            "recommendation": f"Accélération GPU recommandée (speedup estimé: {speedup}x)"
        }
    
    def log_gpu_status(self):
        """Affiche le statut GPU dans les logs"""
        info = self.get_gpu_info()
        perf = self.get_performance_estimate()
        
        if info["recommended_use_gpu"]:
            logger.info(f"🚀 GPU NVIDIA détecté: {info['gpu_count']} GPU(s), {info['gpu_memory']}MB VRAM")
            logger.info(f"✅ Accélération GPU activée (speedup estimé: {perf['expected_speedup']}x)")
        else:
            logger.info("💻 Mode CPU activé (GPU non disponible ou non recommandé)")
            
            if not info["nvidia_gpu"]:
                logger.info("  → Aucune GPU NVIDIA détectée")
            elif not info["cuda_available"]:
                logger.info("  → CUDA non disponible")
            elif not info["paddle_gpu_support"]:
                logger.info("  → PaddlePaddle GPU non supporté")
            elif info["gpu_memory"] <= 2000:
                logger.info(f"  → Mémoire GPU insuffisante ({info['gpu_memory']}MB < 2GB requis)")