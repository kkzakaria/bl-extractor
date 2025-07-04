#!/usr/bin/env python3
"""
Test d'intégration pour l'accélération GPU PaddleOCR
"""

import asyncio
import logging
import sys
import tempfile
import time
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.gpu_detector import GPUDetector
from src.paddleocr_processor import PaddleOCRProcessor
from src.advanced_extractor import AdvancedBLExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_gpu_detection():
    """Test la détection GPU"""
    logger.info("=== Test de détection GPU ===")
    
    detector = GPUDetector()
    gpu_info = detector.get_gpu_info()
    performance = detector.get_performance_estimate()
    
    print(f"GPU NVIDIA détecté: {gpu_info['nvidia_gpu']}")
    print(f"CUDA disponible: {gpu_info['cuda_available']}")
    print(f"Support Paddle GPU: {gpu_info['paddle_gpu_support']}")
    print(f"Mémoire GPU: {gpu_info['gpu_memory']} MB")
    print(f"Nombre de GPU: {gpu_info['gpu_count']}")
    print(f"Recommandation GPU: {gpu_info['recommended_use_gpu']}")
    print(f"Speedup estimé: {performance['expected_speedup']}x")
    
    return detector.should_use_gpu()

async def test_paddleocr_performance():
    """Test les performances PaddleOCR CPU vs GPU"""
    logger.info("\n=== Test de performance PaddleOCR ===")
    
    processor = PaddleOCRProcessor()
    
    if not processor.is_available():
        logger.error("PaddleOCR non disponible")
        return
    
    # Créer une image de test
    import cv2
    import numpy as np
    
    # Image de test avec du texte
    test_image = np.ones((400, 800, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "BILL OF LADING TEST", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(test_image, "B/L NUMBER: BL123456789", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_image, "SHIPPER: ACME SHIPPING CO", (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_image, "CONSIGNEE: DEST COMPANY", (50, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_image, "PORT OF LOADING: HAMBURG", (50, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, test_image)
        
        # Réchauffer le système si GPU disponible
        await processor.warmup_gpu()
        
        # Test d'extraction
        start_time = time.time()
        extracted_text = await processor.extract_text(tmp_file.name)
        extraction_time = time.time() - start_time
        
        # Afficher les résultats
        device = "GPU" if processor.use_gpu else "CPU"
        print(f"\nDevice utilisé: {device}")
        print(f"Temps d'extraction: {extraction_time:.3f}s")
        print(f"Texte extrait ({len(extracted_text)} caractères):")
        print(f"'{extracted_text[:200]}...'")
        
        # Statistiques de performance
        perf_stats = processor.get_performance_summary()
        print(f"\nStatistiques:")
        print(f"  - Extractions totales: {perf_stats['total_extractions']}")
        print(f"  - Temps moyen: {perf_stats['avg_extraction_time']}s")
        print(f"  - Speedup estimé: {perf_stats['estimated_speedup']}x")
        
        # Nettoyer
        import os
        os.unlink(tmp_file.name)

async def test_advanced_extractor_gpu():
    """Test l'extracteur avancé avec support GPU"""
    logger.info("\n=== Test de l'extracteur avancé ===")
    
    extractor = AdvancedBLExtractor()
    
    # Réchauffer le système
    await extractor.warmup_system()
    
    # Vérifier les capacités
    capabilities = extractor.get_capabilities()
    print(f"\nCapacités du système:")
    print(f"  - Accélération GPU: {capabilities.get('gpu_acceleration', False)}")
    print(f"  - Device OCR: {capabilities.get('current_ocr_device', 'Unknown')}")
    print(f"  - Mémoire GPU: {capabilities.get('gpu_memory_mb', 0)} MB")
    print(f"  - Speedup attendu: {capabilities.get('expected_speedup', 1.0)}x")
    
    # Statistiques de performance
    perf_stats = extractor.get_performance_stats()
    print(f"\nStatistiques de performance:")
    for component, stats in perf_stats.items():
        if isinstance(stats, dict) and 'device' in stats:
            print(f"  - {component}: {stats['device']} "
                  f"({stats.get('total_extractions', 0)} extractions)")

async def test_api_endpoints():
    """Test les nouveaux endpoints API"""
    logger.info("\n=== Test des endpoints API ===")
    
    import requests
    import json
    
    base_url = "http://localhost:8000"
    
    try:
        # Test capabilities
        response = requests.get(f"{base_url}/capabilities", timeout=5)
        if response.status_code == 200:
            caps = response.json()
            print("✅ Endpoint /capabilities accessible")
            print(f"   GPU acceleration: {caps['capabilities'].get('gpu_acceleration', False)}")
        else:
            print("❌ Endpoint /capabilities non accessible")
            
        # Test performance
        response = requests.get(f"{base_url}/performance", timeout=5)
        if response.status_code == 200:
            print("✅ Endpoint /performance accessible")
        else:
            print("❌ Endpoint /performance non accessible")
            
        # Test warmup
        response = requests.post(f"{base_url}/warmup", timeout=10)
        if response.status_code == 200:
            print("✅ Endpoint /warmup accessible")
        else:
            print("❌ Endpoint /warmup non accessible")
            
    except requests.exceptions.ConnectionError:
        print("⚠️ API non démarrée. Lancez 'python main.py' pour tester les endpoints.")
    except Exception as e:
        print(f"❌ Erreur lors du test des endpoints: {e}")

async def main():
    """Test principal"""
    print("🧪 Test d'intégration GPU PaddleOCR")
    print("=" * 50)
    
    # 1. Test détection GPU
    gpu_available = await test_gpu_detection()
    
    # 2. Test performance PaddleOCR
    await test_paddleocr_performance()
    
    # 3. Test extracteur avancé
    await test_advanced_extractor_gpu()
    
    # 4. Test endpoints API
    await test_api_endpoints()
    
    print("\n" + "=" * 50)
    if gpu_available:
        print("🚀 Test terminé - Accélération GPU activée!")
    else:
        print("💻 Test terminé - Mode CPU utilisé")
    print("Consultez les logs pour plus de détails.")

if __name__ == "__main__":
    asyncio.run(main())