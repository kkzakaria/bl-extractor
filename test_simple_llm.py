#!/usr/bin/env python3
"""
Test simple du LLM seul
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm_enhancer import LLMEnhancer

async def test_llm_only():
    """Test uniquement le LLM"""
    
    test_text = """
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
    
    print("🧪 Test du LLM Enhancer")
    print("=" * 40)
    
    # Créer l'enhancer
    enhancer = LLMEnhancer()
    
    # Vérifier la disponibilité
    if not enhancer.is_available():
        print("❌ LLM non disponible")
        return
    
    print("✅ LLM disponible")
    print(f"🤖 Modèle: {enhancer.model_name}")
    
    # Extraire avec le LLM
    print("\n🚀 Extraction en cours...")
    result = await enhancer.enhance_extraction(test_text)
    
    if result:
        print("✅ Extraction réussie!")
        print(f"📊 Confiance: {result.extraction_confidence:.2f}")
        print(f"🔧 Méthode: {result.extraction_method}")
        print(f"📋 B/L: {result.bl_number}")
        print(f"📦 Expéditeur: {result.shipper.name if result.shipper else 'Non trouvé'}")
        print(f"🏢 Destinataire: {result.consignee.name if result.consignee else 'Non trouvé'}")
        print(f"🚢 Port départ: {result.port_of_loading.name if result.port_of_loading else 'Non trouvé'}")
        print(f"🏭 Port arrivée: {result.port_of_discharge.name if result.port_of_discharge else 'Non trouvé'}")
        
        if result.cargo:
            print(f"📦 Marchandises: {result.cargo[0].description}")
            print(f"⚖️ Poids: {result.cargo[0].weight}")
        
        print(f"💰 Fret: {result.freight_terms}")
        print(f"📅 Date: {result.issue_date}")
    else:
        print("❌ Extraction échouée")

if __name__ == "__main__":
    asyncio.run(test_llm_only())