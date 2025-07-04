#!/usr/bin/env python3
"""
Test rapide de l'intégration LLM dans le service d'extraction
"""

import asyncio
import tempfile
import os
from src.extractor import BLExtractor

async def test_llm_integration():
    """Test l'intégration du LLM dans l'extracteur"""
    
    # Créer un PDF de test simple (simulé avec du texte)
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
    
    # Créer un fichier temporaire pour simuler un upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(test_content)
        tmp_file_path = tmp_file.name
    
    try:
        # Créer l'extracteur
        extractor = BLExtractor()
        
        # Lire le contenu du fichier
        with open(tmp_file_path, 'rb') as f:
            file_content = f.read()
        
        print("🧪 Test de l'extraction avec LLM...")
        print("=" * 50)
        
        # Test avec LLM
        print("\n1️⃣ Extraction avec LLM (Gemma3:12b)")
        result_llm = await extractor.extract(
            file_content=file_content,
            filename="test_bl.txt",
            ocr_method="tesseract",
            use_llm=True
        )
        
        print(f"✅ Méthode: {result_llm.extraction_method}")
        print(f"📊 Confiance: {result_llm.extraction_confidence:.2f}")
        print(f"📋 B/L: {result_llm.bl_number}")
        print(f"📦 Expéditeur: {result_llm.shipper.name if result_llm.shipper else 'Non trouvé'}")
        print(f"🏢 Destinataire: {result_llm.consignee.name if result_llm.consignee else 'Non trouvé'}")
        print(f"🚢 Port départ: {result_llm.port_of_loading.name if result_llm.port_of_loading else 'Non trouvé'}")
        print(f"🏭 Port arrivée: {result_llm.port_of_discharge.name if result_llm.port_of_discharge else 'Non trouvé'}")
        
        # Test sans LLM (pour comparaison)
        print("\n2️⃣ Extraction sans LLM (Regex uniquement)")
        result_regex = await extractor.extract(
            file_content=file_content,
            filename="test_bl.txt",
            ocr_method="tesseract",
            use_llm=False
        )
        
        print(f"✅ Méthode: {result_regex.extraction_method}")
        print(f"📊 Confiance: {result_regex.extraction_confidence:.2f}")
        print(f"📋 B/L: {result_regex.bl_number}")
        print(f"📦 Expéditeur: {result_regex.shipper.name if result_regex.shipper else 'Non trouvé'}")
        print(f"🏢 Destinataire: {result_regex.consignee.name if result_regex.consignee else 'Non trouvé'}")
        print(f"🚢 Port départ: {result_regex.port_of_loading.name if result_regex.port_of_loading else 'Non trouvé'}")
        print(f"🏭 Port arrivée: {result_regex.port_of_discharge.name if result_regex.port_of_discharge else 'Non trouvé'}")
        
        # Comparaison
        print("\n🔍 COMPARAISON")
        print("=" * 30)
        print(f"LLM - Confiance: {result_llm.extraction_confidence:.2f}")
        print(f"Regex - Confiance: {result_regex.extraction_confidence:.2f}")
        
        improvement = result_llm.extraction_confidence - result_regex.extraction_confidence
        if improvement > 0:
            print(f"✅ LLM améliore l'extraction de {improvement:.2f} points")
        else:
            print(f"⚠️ Pas d'amélioration notable")
        
        # Recommandation
        if result_llm.extraction_confidence > 0.7:
            print("\n💡 Recommandation: LLM activé par défaut")
        else:
            print("\n⚠️ Recommandation: Vérifier la configuration du LLM")
        
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

if __name__ == "__main__":
    asyncio.run(test_llm_integration())