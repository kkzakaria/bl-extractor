#!/usr/bin/env python3
"""
Comparaison des différentes solutions OCR
"""

import time
import asyncio
from typing import Dict, Any
import tempfile
import cv2
import numpy as np

# OCR Libraries
import pytesseract
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

class OCRBenchmark:
    """Benchmark des différentes solutions OCR"""
    
    def __init__(self):
        self.test_image = self._create_test_image()
        self.ocr_engines = self._init_engines()
    
    def _create_test_image(self):
        """Crée une image de test avec du texte de connaissement"""
        # Créer une image blanche
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Ajouter du texte simulant un connaissement
        text_lines = [
            "BILL OF LADING NO: ABC123456789",
            "BOOKING NO: BK987654321",
            "",
            "SHIPPER: ACME SHIPPING COMPANY",
            "123 MAIN STREET, HAMBURG, GERMANY",
            "",
            "CONSIGNEE: GLOBAL IMPORT CORP", 
            "456 OAK AVENUE, NEW YORK, USA",
            "",
            "PORT OF LOADING: HAMBURG, GERMANY",
            "PORT OF DISCHARGE: NEW YORK, USA",
            "",
            "VESSEL: EVER GIVEN",
            "VOYAGE: V001",
            "",
            "DESCRIPTION: GENERAL MERCHANDISE",
            "QUANTITY: 100 CARTONS",
            "WEIGHT: 2500 KG"
        ]
        
        # Utiliser OpenCV pour ajouter le texte
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        for line in text_lines:
            if line:  # Skip empty lines
                cv2.putText(img, line, (20, y_offset), font, 0.6, (0, 0, 0), 1)
            y_offset += 25
        
        return img
    
    def _init_engines(self):
        """Initialise les moteurs OCR disponibles"""
        engines = {}
        
        # Tesseract (toujours disponible)
        engines['tesseract'] = True
        
        # EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                engines['easyocr'] = easyocr.Reader(['en'])
                print("✅ EasyOCR initialisé")
            except Exception as e:
                print(f"❌ EasyOCR échec: {e}")
                engines['easyocr'] = False
        else:
            engines['easyocr'] = False
            print("❌ EasyOCR non installé")
        
        # PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                engines['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                print("✅ PaddleOCR initialisé")
            except Exception as e:
                print(f"❌ PaddleOCR échec: {e}")
                engines['paddleocr'] = False
        else:
            engines['paddleocr'] = False
            print("❌ PaddleOCR non installé")
        
        return engines
    
    async def benchmark_tesseract(self, image_path: str) -> Dict[str, Any]:
        """Benchmark Tesseract"""
        try:
            start_time = time.time()
            
            # Configuration Tesseract
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(
                image_path,
                config=custom_config,
                lang='eng'
            )
            
            end_time = time.time()
            
            return {
                'engine': 'Tesseract',
                'text': text.strip(),
                'time': end_time - start_time,
                'success': True,
                'words_extracted': len(text.split())
            }
            
        except Exception as e:
            return {
                'engine': 'Tesseract',
                'text': '',
                'time': 0,
                'success': False,
                'error': str(e),
                'words_extracted': 0
            }
    
    async def benchmark_easyocr(self, image_path: str) -> Dict[str, Any]:
        """Benchmark EasyOCR"""
        if not self.ocr_engines.get('easyocr'):
            return {
                'engine': 'EasyOCR',
                'text': '',
                'time': 0,
                'success': False,
                'error': 'Not available',
                'words_extracted': 0
            }
        
        try:
            start_time = time.time()
            
            results = self.ocr_engines['easyocr'].readtext(image_path)
            text = ' '.join([result[1] for result in results])
            
            end_time = time.time()
            
            return {
                'engine': 'EasyOCR',
                'text': text.strip(),
                'time': end_time - start_time,
                'success': True,
                'words_extracted': len(text.split())
            }
            
        except Exception as e:
            return {
                'engine': 'EasyOCR',
                'text': '',
                'time': 0,
                'success': False,
                'error': str(e),
                'words_extracted': 0
            }
    
    async def benchmark_paddleocr(self, image_path: str) -> Dict[str, Any]:
        """Benchmark PaddleOCR"""
        if not self.ocr_engines.get('paddleocr'):
            return {
                'engine': 'PaddleOCR',
                'text': '',
                'time': 0,
                'success': False,
                'error': 'Not available',
                'words_extracted': 0
            }
        
        try:
            start_time = time.time()
            
            results = self.ocr_engines['paddleocr'].ocr(image_path, cls=True)
            text_lines = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) > 1:
                        text_lines.append(line[1][0])
            
            text = ' '.join(text_lines)
            end_time = time.time()
            
            return {
                'engine': 'PaddleOCR',
                'text': text.strip(),
                'time': end_time - start_time,
                'success': True,
                'words_extracted': len(text.split())
            }
            
        except Exception as e:
            return {
                'engine': 'PaddleOCR',
                'text': '',
                'time': 0,
                'success': False,
                'error': str(e),
                'words_extracted': 0
            }
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """Exécute le benchmark complet"""
        # Sauvegarder l'image de test
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, self.test_image)
            image_path = tmp_file.name
        
        print("🔍 BENCHMARK OCR - EXTRACTION DE CONNAISSEMENTS")
        print("=" * 60)
        
        results = []
        
        # Test de chaque moteur
        print("\n📊 Tests en cours...")
        
        # Tesseract
        print("• Tesseract...")
        tesseract_result = await self.benchmark_tesseract(image_path)
        results.append(tesseract_result)
        
        # EasyOCR
        print("• EasyOCR...")
        easyocr_result = await self.benchmark_easyocr(image_path)
        results.append(easyocr_result)
        
        # PaddleOCR
        print("• PaddleOCR...")
        paddleocr_result = await self.benchmark_paddleocr(image_path)
        results.append(paddleocr_result)
        
        # Nettoyer
        import os
        os.unlink(image_path)
        
        return results
    
    def generate_report(self, results):
        """Génère le rapport de benchmark"""
        print("\n📋 RÉSULTATS")
        print("=" * 40)
        
        # Trier par nombre de mots extraits
        successful_results = [r for r in results if r['success']]
        successful_results.sort(key=lambda x: x['words_extracted'], reverse=True)
        
        for i, result in enumerate(successful_results, 1):
            print(f"\n{i}. {result['engine']}")
            print(f"   ⏱️  Temps: {result['time']:.2f}s")
            print(f"   📝 Mots extraits: {result['words_extracted']}")
            print(f"   ✅ Statut: {'Succès' if result['success'] else 'Échec'}")
            
            # Afficher un échantillon du texte
            if result['text']:
                sample = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                print(f"   📄 Échantillon: {sample}")
        
        # Échecs
        failed_results = [r for r in results if not r['success']]
        if failed_results:
            print(f"\n❌ ÉCHECS:")
            for result in failed_results:
                print(f"   • {result['engine']}: {result.get('error', 'Erreur inconnue')}")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        print("=" * 30)
        
        if successful_results:
            best = successful_results[0]
            print(f"🥇 Meilleur: {best['engine']}")
            print(f"   Vitesse: {best['time']:.1f}s")
            print(f"   Précision: {best['words_extracted']} mots")
            
            # Recommandations spécifiques
            if best['engine'] == 'PaddleOCR':
                print("\n📦 Installation PaddleOCR:")
                print("   pip install paddlepaddle paddleocr")
            elif best['engine'] == 'EasyOCR':
                print("\n📦 Installation EasyOCR:")
                print("   pip install easyocr")
        
        print(f"\n🏭 Pour la production:")
        print("   • AWS Textract (meilleure précision)")
        print("   • Google Vision API (excellent pour documents)")
        print("   • Azure Read API (bonne alternative)")

async def main():
    """Fonction principale"""
    benchmark = OCRBenchmark()
    results = await benchmark.run_benchmark()
    benchmark.generate_report(results)

if __name__ == "__main__":
    asyncio.run(main())