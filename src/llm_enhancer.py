import logging
import json
import ollama
from typing import Optional, Dict, Any
from .models import BillOfLadingData, Party, Port, Cargo, Container, TransportDetails

logger = logging.getLogger(__name__)

class LLMEnhancer:
    """Améliorateur LLM pour l'extraction de données de connaissements"""
    
    def __init__(self, model_name: str = "gemma3:12b"):
        self.model_name = model_name
        self.confidence_threshold = 0.8
    
    async def enhance_extraction(self, raw_text: str, structured_data: Optional[Dict[str, Any]] = None) -> Optional[BillOfLadingData]:
        """
        Améliore l'extraction en utilisant un LLM
        
        Args:
            raw_text: Texte brut extrait par OCR
            structured_data: Données structurées de Docling (optionnel)
            
        Returns:
            BillOfLadingData: Données extraites et structurées
        """
        try:
            logger.info(f"Amélioration de l'extraction avec {self.model_name}")
            
            # Créer le prompt optimisé
            prompt = self._create_extraction_prompt(raw_text, structured_data)
            
            # Exécuter l'extraction LLM
            llm_result = await self._query_llm(prompt)
            
            if not llm_result:
                return None
            
            # Convertir en BillOfLadingData
            enhanced_data = self._convert_to_bill_of_lading(llm_result)
            
            # Marquer comme extraction LLM
            enhanced_data.extraction_method = f"llm_{self.model_name}"
            enhanced_data.raw_text = raw_text
            
            # Calculer la confiance
            enhanced_data.extraction_confidence = self._calculate_llm_confidence(enhanced_data)
            
            logger.info(f"Extraction LLM terminée avec confiance: {enhanced_data.extraction_confidence}")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Erreur lors de l'amélioration LLM: {str(e)}")
            return None
    
    def _create_extraction_prompt(self, text: str, structured_data: Optional[Dict[str, Any]] = None) -> str:
        """Crée un prompt optimisé pour l'extraction"""
        
        # Prompt de base
        prompt = f"""
        Extrait les données de ce connaissement (Bill of Lading) et retourne UNIQUEMENT un JSON valide.
        
        Texte du connaissement:
        {text}"""
        
        # Ajouter les données structurées si disponibles
        if structured_data:
            prompt += f"""
        
        Données structurées additionnelles (Docling):
        {json.dumps(structured_data, indent=2, ensure_ascii=False)}
        
        IMPORTANT: Utilise ces données structurées pour améliorer la précision de l'extraction.
        Les sections identifiées peuvent t'aider à localiser les bonnes informations."""
        
        prompt += """
        
        Retourne un JSON avec ces champs (utilise null si non trouvé):
        {{
            "bl_number": "numéro du connaissement",
            "booking_number": "numéro de réservation",
            "shipper_name": "nom de l'expéditeur",
            "shipper_address": "adresse complète de l'expéditeur",
            "consignee_name": "nom du destinataire",
            "consignee_address": "adresse complète du destinataire",
            "notify_party_name": "nom de la partie à notifier",
            "notify_party_address": "adresse de la partie à notifier",
            "port_of_loading": "port de chargement",
            "port_of_discharge": "port de déchargement",
            "port_of_delivery": "port de livraison",
            "vessel_name": "nom du navire",
            "voyage_number": "numéro de voyage",
            "departure_date": "date de départ",
            "arrival_date": "date d'arrivée",
            "cargo_description": "description des marchandises",
            "quantity": "quantité",
            "weight": "poids",
            "volume": "volume",
            "container_numbers": ["liste des numéros de conteneurs"],
            "freight_terms": "conditions de fret (PREPAID/COLLECT)",
            "issue_date": "date d'émission",
            "issue_place": "lieu d'émission"
        }}
        
        Règles importantes:
        - Retourne UNIQUEMENT le JSON, pas de texte avant/après
        - Utilise null pour les champs non trouvés
        - Corrige les erreurs OCR évidentes (0->O, 1->I, etc.)
        - Garde le format des dates tel qu'écrit
        """
    
    async def _query_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Interroge le LLM et parse la réponse"""
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Faible température pour plus de cohérence
                    "top_p": 0.9,
                    "top_k": 10
                }
            )
            
            result_text = response['response'].strip()
            
            # Nettoyer la réponse
            result_text = self._clean_llm_response(result_text)
            
            # Parser le JSON
            parsed_result = json.loads(result_text)
            
            return parsed_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur JSON LLM: {str(e)}")
            logger.error(f"Réponse brute: {result_text[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Erreur requête LLM: {str(e)}")
            return None
    
    def _clean_llm_response(self, response: str) -> str:
        """Nettoie la réponse du LLM pour extraire le JSON"""
        # Supprimer les blocs de code markdown
        if response.startswith('```'):
            lines = response.split('\n')
            # Trouver le début et la fin du JSON
            start_idx = 0
            end_idx = len(lines)
            
            for i, line in enumerate(lines):
                if line.strip().startswith('{'):
                    start_idx = i
                    break
            
            for i in range(len(lines) - 1, -1, -1):
                if line.strip().endswith('}'):
                    end_idx = i + 1
                    break
            
            response = '\n'.join(lines[start_idx:end_idx])
        
        # Supprimer les textes avant/après le JSON
        if '{' in response and '}' in response:
            start = response.find('{')
            end = response.rfind('}') + 1
            response = response[start:end]
        
        return response.strip()
    
    def _convert_to_bill_of_lading(self, llm_data: Dict[str, Any]) -> BillOfLadingData:
        """Convertit les données LLM en BillOfLadingData"""
        # Créer l'objet principal
        bill_data = BillOfLadingData()
        
        # Numéros de référence
        bill_data.bl_number = llm_data.get('bl_number')
        bill_data.booking_number = llm_data.get('booking_number')
        
        # Parties
        if llm_data.get('shipper_name'):
            bill_data.shipper = Party(
                name=llm_data.get('shipper_name'),
                address=llm_data.get('shipper_address')
            )
        
        if llm_data.get('consignee_name'):
            bill_data.consignee = Party(
                name=llm_data.get('consignee_name'),
                address=llm_data.get('consignee_address')
            )
        
        if llm_data.get('notify_party_name'):
            bill_data.notify_party = Party(
                name=llm_data.get('notify_party_name'),
                address=llm_data.get('notify_party_address')
            )
        
        # Ports
        if llm_data.get('port_of_loading'):
            bill_data.port_of_loading = Port(name=llm_data.get('port_of_loading'))
        
        if llm_data.get('port_of_discharge'):
            bill_data.port_of_discharge = Port(name=llm_data.get('port_of_discharge'))
        
        if llm_data.get('port_of_delivery'):
            bill_data.port_of_delivery = Port(name=llm_data.get('port_of_delivery'))
        
        # Détails de transport
        if any([llm_data.get('vessel_name'), llm_data.get('voyage_number')]):
            bill_data.transport_details = TransportDetails(
                vessel_name=llm_data.get('vessel_name'),
                voyage_number=llm_data.get('voyage_number'),
                bl_number=llm_data.get('bl_number'),
                booking_number=llm_data.get('booking_number'),
                departure_date=llm_data.get('departure_date'),
                arrival_date=llm_data.get('arrival_date')
            )
        
        # Marchandises
        if llm_data.get('cargo_description'):
            cargo = Cargo(
                description=llm_data.get('cargo_description'),
                quantity=llm_data.get('quantity'),
                weight=llm_data.get('weight'),
                volume=llm_data.get('volume')
            )
            bill_data.cargo = [cargo]
        
        # Conteneurs
        container_numbers = llm_data.get('container_numbers', [])
        if container_numbers:
            bill_data.containers = [
                Container(number=num) for num in container_numbers if num
            ]
        
        # Conditions
        bill_data.freight_terms = llm_data.get('freight_terms')
        bill_data.issue_date = llm_data.get('issue_date')
        
        return bill_data
    
    def _calculate_llm_confidence(self, data: BillOfLadingData) -> float:
        """Calcule la confiance de l'extraction LLM"""
        total_fields = 0
        filled_fields = 0
        
        # Champs critiques
        critical_fields = [
            data.bl_number,
            data.shipper,
            data.consignee,
            data.port_of_loading,
            data.port_of_discharge
        ]
        
        for field in critical_fields:
            total_fields += 1
            if field:
                filled_fields += 1
        
        # Champs importants
        important_fields = [
            data.booking_number,
            data.transport_details,
            data.cargo,
            data.freight_terms
        ]
        
        for field in important_fields:
            total_fields += 0.5
            if field:
                filled_fields += 0.5
        
        # Bonus pour les conteneurs
        if data.containers:
            filled_fields += 0.2
        total_fields += 0.2
        
        if total_fields == 0:
            return 0.0
        
        confidence = filled_fields / total_fields
        
        # Bonus LLM (généralement plus fiable que regex)
        confidence = min(confidence * 1.1, 1.0)
        
        return confidence
    
    def is_available(self) -> bool:
        """Vérifie si le modèle LLM est disponible"""
        try:
            ollama.list()
            return True
        except Exception:
            return False