import asyncio
import json
import time
from typing import Dict, List, Any
import ollama

class LLMBenchmark:
    """Benchmark pour comparer les modèles LLM sur l'extraction de connaissements"""
    
    def __init__(self):
        self.models = ["qwen2.5vl:32b", "gemma3:12b"]
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """Crée des cas de test pour le benchmark"""
        return [
            {
                "name": "Connaissement Standard",
                "text": """
                BILL OF LADING NO: ABCD1234567890
                BOOKING NO: BK987654321
                SHIPPER: ACME SHIPPING COMPANY
                123 MAIN STREET, HAMBURG, GERMANY
                CONSIGNEE: GLOBAL IMPORT CORP
                456 OAK AVENUE, NEW YORK, USA
                NOTIFY PARTY: SAME AS CONSIGNEE
                PORT OF LOADING: HAMBURG, GERMANY
                PORT OF DISCHARGE: NEW YORK, USA
                VESSEL: EVER GIVEN
                VOYAGE: V001
                DESCRIPTION OF GOODS: GENERAL MERCHANDISE
                QUANTITY: 100 CARTONS
                GROSS WEIGHT: 2500 KG
                MEASUREMENT: 150 CBM
                FREIGHT: PREPAID
                PLACE AND DATE OF ISSUE: HAMBURG, 15/01/2024
                CONTAINER NO: ABCD1234567
                """,
                "expected_fields": [
                    "bl_number", "booking_number", "shipper", "consignee", 
                    "port_of_loading", "port_of_discharge", "vessel_name", 
                    "voyage_number", "cargo_description", "freight_terms"
                ]
            },
            {
                "name": "Connaissement Français",
                "text": """
                CONNAISSEMENT NO: FR789012345
                NUMERO DE RESERVATION: RES456789
                EXPEDITEUR: SOCIETE MARITIME FRANCAISE
                12 RUE DU PORT, MARSEILLE, FRANCE
                DESTINATAIRE: IMPORT CANADA INC
                789 MAPLE STREET, MONTREAL, CANADA
                PARTIE A NOTIFIER: MEME QUE DESTINATAIRE
                PORT DE CHARGEMENT: MARSEILLE, FRANCE
                PORT DE DECHARGEMENT: MONTREAL, CANADA
                NAVIRE: ATLANTIC STAR
                VOYAGE: AS2024
                DESCRIPTION DES MARCHANDISES: PRODUITS ALIMENTAIRES
                QUANTITE: 50 PALETTES
                POIDS BRUT: 1200 KG
                VOLUME: 80 M3
                FRET: PREPAYE
                LIEU ET DATE D'EMISSION: MARSEILLE, 20/02/2024
                CONTENEUR NO: FREN9876543
                """,
                "expected_fields": [
                    "bl_number", "booking_number", "shipper", "consignee",
                    "port_of_loading", "port_of_discharge", "vessel_name",
                    "voyage_number", "cargo_description", "freight_terms"
                ]
            },
            {
                "name": "Connaissement avec OCR Errors",
                "text": """
                B1LL 0F LAD1NG N0: XYZ987654321
                B00K1NG N0: BK12345O
                SH1PPER: 0CEAN CARR1ER LTD
                456 WATER STR33T, L0ND0N, UK
                C0NS1GN33: DEST1NAT10N C0RP
                123 1NDUSTR1AL AVE, T0KY0, JAPAN
                P0RT 0F L0AD1NG: L0ND0N, UK
                P0RT 0F D1SCHARGE: T0KY0, JAPAN
                VESS3L: PAC1F1C DREAM
                V0YAG3: PD2024
                D3SCR1PT10N: M1XED C0NTA1N3R L0AD
                QU4NT1TY: 1 C0NTA1N3R
                WE1GHT: 15000 KG
                """,
                "expected_fields": [
                    "bl_number", "booking_number", "shipper", "consignee",
                    "port_of_loading", "port_of_discharge", "vessel_name",
                    "voyage_number", "cargo_description"
                ]
            }
        ]
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """Exécute le benchmark complet"""
        results = {}
        
        for model in self.models:
            print(f"\n🔍 Test du modèle: {model}")
            print("=" * 50)
            
            model_results = {
                "model": model,
                "test_results": [],
                "overall_score": 0.0,
                "average_time": 0.0,
                "total_tests": len(self.test_cases),
                "successful_extractions": 0
            }
            
            total_time = 0
            total_score = 0
            
            for i, test_case in enumerate(self.test_cases):
                print(f"\n📋 Test {i+1}: {test_case['name']}")
                
                # Mesurer le temps d'exécution
                start_time = time.time()
                
                try:
                    # Exécuter l'extraction
                    result = await self._extract_with_model(model, test_case['text'])
                    execution_time = time.time() - start_time
                    
                    # Évaluer les résultats
                    score = self._evaluate_result(result, test_case['expected_fields'])
                    
                    test_result = {
                        "test_name": test_case['name'],
                        "execution_time": execution_time,
                        "score": score,
                        "extracted_data": result,
                        "success": score > 0.3  # Seuil de succès
                    }
                    
                    model_results["test_results"].append(test_result)
                    
                    total_time += execution_time
                    total_score += score
                    
                    if test_result["success"]:
                        model_results["successful_extractions"] += 1
                    
                    print(f"  ⏱️  Temps: {execution_time:.2f}s")
                    print(f"  📊 Score: {score:.2f}")
                    print(f"  ✅ Succès: {'Oui' if test_result['success'] else 'Non'}")
                    
                except Exception as e:
                    print(f"  ❌ Erreur: {str(e)}")
                    test_result = {
                        "test_name": test_case['name'],
                        "execution_time": 0,
                        "score": 0,
                        "extracted_data": None,
                        "success": False,
                        "error": str(e)
                    }
                    model_results["test_results"].append(test_result)
            
            # Calculer les moyennes
            if len(self.test_cases) > 0:
                model_results["overall_score"] = total_score / len(self.test_cases)
                model_results["average_time"] = total_time / len(self.test_cases)
            
            results[model] = model_results
        
        return results
    
    async def _extract_with_model(self, model: str, text: str) -> Dict[str, Any]:
        """Extrait les données avec un modèle spécifique"""
        prompt = f"""
        Extrait les données de ce connaissement (Bill of Lading) et retourne-les au format JSON.
        
        Texte du connaissement:
        {text}
        
        Retourne UNIQUEMENT un objet JSON valide avec ces champs possibles:
        - bl_number: numéro du connaissement
        - booking_number: numéro de réservation
        - shipper: expéditeur (nom et adresse)
        - consignee: destinataire (nom et adresse)
        - notify_party: partie à notifier
        - port_of_loading: port de chargement
        - port_of_discharge: port de déchargement
        - vessel_name: nom du navire
        - voyage_number: numéro de voyage
        - cargo_description: description des marchandises
        - quantity: quantité
        - weight: poids
        - volume: volume
        - freight_terms: conditions de fret
        - issue_date: date d'émission
        - container_number: numéro de conteneur
        
        Exemple de format attendu:
        {{
            "bl_number": "ABCD1234567890",
            "shipper": "ACME SHIPPING COMPANY, 123 MAIN STREET, HAMBURG, GERMANY",
            "consignee": "GLOBAL IMPORT CORP, 456 OAK AVENUE, NEW YORK, USA"
        }}
        """
        
        try:
            response = ollama.generate(model=model, prompt=prompt)
            result_text = response['response'].strip()
            
            # Nettoyer la réponse pour extraire le JSON
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            # Parser le JSON
            return json.loads(result_text)
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️  Erreur JSON: {str(e)}")
            return {"error": f"Invalid JSON: {str(e)}"}
        except Exception as e:
            print(f"  ❌ Erreur modèle: {str(e)}")
            return {"error": str(e)}
    
    def _evaluate_result(self, result: Dict[str, Any], expected_fields: List[str]) -> float:
        """Évalue la qualité de l'extraction"""
        if not result or "error" in result:
            return 0.0
        
        extracted_fields = 0
        total_fields = len(expected_fields)
        
        for field in expected_fields:
            if field in result and result[field] and result[field].strip():
                extracted_fields += 1
        
        return extracted_fields / total_fields if total_fields > 0 else 0.0
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Génère un rapport de benchmark"""
        report = "\n" + "=" * 60
        report += "\n🏆 RAPPORT DE BENCHMARK - EXTRACTION DE CONNAISSEMENTS"
        report += "\n" + "=" * 60
        
        # Classement global
        models_sorted = sorted(
            results.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        report += "\n\n📊 CLASSEMENT GÉNÉRAL:"
        report += "\n" + "-" * 30
        
        for rank, (model, data) in enumerate(models_sorted, 1):
            report += f"\n{rank}. {model}"
            report += f"\n   Score global: {data['overall_score']:.2f}"
            report += f"\n   Temps moyen: {data['average_time']:.2f}s"
            report += f"\n   Extractions réussies: {data['successful_extractions']}/{data['total_tests']}"
            report += f"\n   Taux de succès: {(data['successful_extractions']/data['total_tests']*100):.1f}%"
            report += "\n"
        
        # Détails par modèle
        for model, data in results.items():
            report += f"\n\n📋 DÉTAILS - {model}:"
            report += "\n" + "-" * 40
            
            for test in data['test_results']:
                report += f"\n• {test['test_name']}: "
                report += f"Score {test['score']:.2f}, "
                report += f"Temps {test['execution_time']:.2f}s"
                if not test['success']:
                    report += " ❌"
                else:
                    report += " ✅"
        
        # Recommandations
        report += "\n\n💡 RECOMMANDATIONS:"
        report += "\n" + "-" * 20
        
        winner = models_sorted[0]
        report += f"\n🥇 Meilleur modèle: {winner[0]}"
        report += f"\n   - Score: {winner[1]['overall_score']:.2f}"
        report += f"\n   - Temps: {winner[1]['average_time']:.2f}s"
        
        # Analyse comparative
        if len(models_sorted) > 1:
            report += "\n\n🔍 ANALYSE COMPARATIVE:"
            model1, data1 = models_sorted[0]
            model2, data2 = models_sorted[1]
            
            if data1['overall_score'] > data2['overall_score']:
                diff = data1['overall_score'] - data2['overall_score']
                report += f"\n• {model1} est {diff:.1%} plus précis que {model2}"
            
            if data1['average_time'] < data2['average_time']:
                diff = data2['average_time'] - data1['average_time']
                report += f"\n• {model1} est {diff:.1f}s plus rapide que {model2}"
        
        return report

async def main():
    """Fonction principale pour exécuter le benchmark"""
    benchmark = LLMBenchmark()
    
    print("🚀 Démarrage du benchmark LLM pour l'extraction de connaissements...")
    print("📝 Modèles à tester: qwen2.5vl:32b, gemma3:12b")
    print("🎯 Cas de test: Connaissements standard, français, avec erreurs OCR")
    
    # Exécuter le benchmark
    results = await benchmark.run_benchmark()
    
    # Générer et afficher le rapport
    report = benchmark.generate_report(results)
    print(report)
    
    # Sauvegarder les résultats
    with open('/home/super/Codes/bl-extractor/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Résultats sauvegardés dans: benchmark_results.json")

if __name__ == "__main__":
    asyncio.run(main())