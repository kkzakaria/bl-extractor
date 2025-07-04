#!/usr/bin/env python3
"""
Test de l'intégration complète Docling + LLM
"""

import asyncio
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.advanced_extractor import AdvancedBLExtractor

async def test_advanced_extraction():
    """Test l'extraction avancée avec Docling + LLM"""
    
    # Créer un contenu de test simple
    test_content = """
    BILL OF LADING NO: TEST123456789
    BOOKING NO: BK987654321
    
    SHIPPER: ACME SHIPPING COMPANY
    123 MAIN STREET, HAMBURG, GERMANY
    
    CONSIGNEE: GLOBAL IMPORT CORP
    456 OAK AVENUE, NEW YORK, USA
    
    PORT OF LOADING: HAMBURG, GERMANY
    PORT OF DISCHARGE: NEW YORK, USA
    
    VESSEL: EVER GIVEN
    VOYAGE: V001
    
    DESCRIPTION OF GOODS: GENERAL MERCHANDISE
    QUANTITY: 100 CARTONS
    GROSS WEIGHT: 2500 KG
    
    FREIGHT: PREPAID
    PLACE AND DATE OF ISSUE: HAMBURG, 15/01/2024
    """
    
    # Créer un fichier temporaire
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(test_content)
        tmp_file_path = tmp_file.name
    
    try:
        print("🚀 Test de l'Extraction Avancée (Docling + LLM)")
        print("=" * 60)
        
        # Créer l'extracteur avancé
        extractor = AdvancedBLExtractor()
        
        # Vérifier les capacités
        capabilities = extractor.get_capabilities()
        print("\n📊 CAPACITÉS DU SERVICE:")
        for capability, available in capabilities.items():
            status = "✅" if available else "❌"
            print(f"   {status} {capability}")
        
        # Recommandations par type de fichier
        print("\n💡 STRATÉGIES RECOMMANDÉES:")
        print(f"   📄 PDF: {extractor.get_recommended_strategy('.pdf')}")
        print(f"   🖼️  Image: {extractor.get_recommended_strategy('.jpg')}")
        
        # Lire le contenu du fichier
        with open(tmp_file_path, 'rb') as f:
            file_content = f.read()
        
        print(f"\n🧪 TEST D'EXTRACTION:")
        print("-" * 40)
        
        # Test avec toutes les options activées
        print("\n1️⃣ Extraction complète (Docling + LLM)")
        result_full = await extractor.extract(
            file_content=file_content,
            filename="test_bl.txt",
            ocr_method="tesseract",
            use_llm=True,
            use_docling=True
        )
        
        print(f"   📊 Méthode: {result_full.extraction_method}")
        print(f"   🎯 Confiance: {result_full.extraction_confidence:.2f}")
        print(f"   📋 B/L: {result_full.bl_number}")
        print(f"   📦 Expéditeur: {result_full.shipper.name if result_full.shipper else 'Non trouvé'}")
        print(f"   🏢 Destinataire: {result_full.consignee.name if result_full.consignee else 'Non trouvé'}")
        
        # Test sans Docling
        print("\n2️⃣ Extraction sans Docling (LLM uniquement)")
        result_no_docling = await extractor.extract(
            file_content=file_content,
            filename="test_bl.txt",
            ocr_method="tesseract",
            use_llm=True,
            use_docling=False
        )
        
        print(f"   📊 Méthode: {result_no_docling.extraction_method}")
        print(f"   🎯 Confiance: {result_no_docling.extraction_confidence:.2f}")
        
        # Test basique (regex seulement)
        print("\n3️⃣ Extraction basique (Regex uniquement)")
        result_basic = await extractor.extract(
            file_content=file_content,
            filename="test_bl.txt",
            ocr_method="tesseract",
            use_llm=False,
            use_docling=False
        )
        
        print(f"   📊 Méthode: {result_basic.extraction_method}")
        print(f"   🎯 Confiance: {result_basic.extraction_confidence:.2f}")
        
        # Comparaison
        print(f"\n📈 COMPARAISON DES PERFORMANCES:")
        print("=" * 50)
        
        results = [
            ("Docling + LLM", result_full),
            ("LLM seul", result_no_docling),
            ("Regex seul", result_basic)
        ]
        
        for name, result in sorted(results, key=lambda x: x[1].extraction_confidence, reverse=True):
            print(f"🏆 {name}: {result.extraction_confidence:.2f} ({result.extraction_method})")
        
        # Recommandations finales
        best_result = max(results, key=lambda x: x[1].extraction_confidence)
        print(f"\n💡 RECOMMANDATION:")
        print(f"   Meilleure méthode: {best_result[0]}")
        print(f"   Confiance: {best_result[1].extraction_confidence:.2f}")
        print(f"   Méthode technique: {best_result[1].extraction_method}")
        
        if capabilities["docling_available"] and capabilities["llm_available"]:
            print(f"\n✅ STACK OPTIMALE DISPONIBLE!")
            print(f"   Docling + Gemma3:12b = Précision maximale")
        elif capabilities["llm_available"]:
            print(f"\n⚠️ Docling manquant, mais LLM disponible")
            print(f"   Installation: pip install docling")
        else:
            print(f"\n❌ Stack basique uniquement")
            print(f"   Amélioration possible avec LLM + Docling")
        
    finally:
        # Nettoyer
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

if __name__ == "__main__":
    asyncio.run(test_advanced_extraction())