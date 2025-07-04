#!/usr/bin/env python3
import ollama
import json
import time

def test_model(model_name, test_text):
    """Test rapide d'un modèle sur l'extraction"""
    
    prompt = f"""
    Extrait les données de ce connaissement et retourne UNIQUEMENT un JSON valide:
    
    {test_text}
    
    Format JSON attendu:
    {{
        "bl_number": "...",
        "shipper": "...",
        "consignee": "...",
        "port_of_loading": "...",
        "port_of_discharge": "..."
    }}
    """
    
    print(f"\n🔍 Test du modèle: {model_name}")
    print("-" * 40)
    
    try:
        start_time = time.time()
        response = ollama.generate(model=model_name, prompt=prompt)
        end_time = time.time()
        
        result = response['response'].strip()
        
        # Essayer de parser le JSON
        try:
            parsed = json.loads(result)
            success = True
            error = None
        except json.JSONDecodeError as e:
            # Essayer de nettoyer le JSON
            if result.startswith('```'):
                result = result.split('```')[1]
                if result.startswith('json'):
                    result = result[4:].strip()
            try:
                parsed = json.loads(result)
                success = True
                error = None
            except:
                success = False
                error = str(e)
                parsed = None
        
        print(f"⏱️  Temps d'exécution: {end_time - start_time:.2f}s")
        print(f"✅ JSON valide: {'Oui' if success else 'Non'}")
        
        if success:
            fields_found = len([k for k, v in parsed.items() if v and str(v).strip()])
            print(f"📊 Champs extraits: {fields_found}/5")
            print(f"🎯 Score: {fields_found/5:.1%}")
            
            # Afficher quelques champs clés
            if 'bl_number' in parsed:
                print(f"📋 B/L: {parsed['bl_number']}")
            if 'shipper' in parsed:
                print(f"📦 Expéditeur: {parsed['shipper'][:50]}...")
        else:
            print(f"❌ Erreur: {error}")
            print(f"🔍 Réponse brute: {result[:100]}...")
        
        return {
            'model': model_name,
            'time': end_time - start_time,
            'success': success,
            'fields_found': fields_found if success else 0,
            'score': fields_found/5 if success else 0,
            'result': parsed if success else None
        }
        
    except Exception as e:
        print(f"❌ Erreur avec le modèle: {str(e)}")
        return {
            'model': model_name,
            'time': 0,
            'success': False,
            'fields_found': 0,
            'score': 0,
            'error': str(e)
        }

def main():
    # Test case simple
    test_text = """
    BILL OF LADING NO: ABCD1234567890
    SHIPPER: ACME SHIPPING COMPANY
    123 MAIN STREET, HAMBURG, GERMANY
    CONSIGNEE: GLOBAL IMPORT CORP
    456 OAK AVENUE, NEW YORK, USA
    PORT OF LOADING: HAMBURG, GERMANY
    PORT OF DISCHARGE: NEW YORK, USA
    VESSEL: EVER GIVEN
    VOYAGE: V001
    """
    
    print("🚀 Test rapide des modèles LLM")
    print("=" * 50)
    
    models = ["qwen2.5vl:32b", "gemma3:12b"]
    results = []
    
    for model in models:
        result = test_model(model, test_text)
        results.append(result)
    
    # Comparaison
    print("\n\n🏆 COMPARAISON")
    print("=" * 30)
    
    for result in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"{result['model']}: Score {result['score']:.1%}, Temps {result['time']:.1f}s")
    
    # Recommandation
    best = max(results, key=lambda x: x['score'])
    print(f"\n💡 Recommandation: {best['model']}")
    print(f"   Score: {best['score']:.1%}")
    print(f"   Temps: {best['time']:.1f}s")

if __name__ == "__main__":
    main()