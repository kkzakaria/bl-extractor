import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from .models import BillOfLadingData, Party, Port, Cargo, Container, TransportDetails

logger = logging.getLogger(__name__)

class TextParser:
    """Parseur pour extraire les données structurées depuis le texte OCR"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialise les patterns de regex pour l'extraction"""
        
        # Patterns pour les numéros de référence
        self.patterns = {
            'bl_number': [
                r'(?:B/L|BL|BILL OF LADING)[\s\w]*:?\s*([A-Z0-9]{8,20})',
                r'(?:BILL OF LADING|BL)\s*(?:NO|NUMBER|#):?\s*([A-Z0-9]{8,20})',
                r'(?:CONNAISSEMENT|CONNAISSANCE)[\s\w]*:?\s*([A-Z0-9]{8,20})'
            ],
            'booking_number': [
                r'(?:BOOKING|RÉSERVATION)[\s\w]*:?\s*([A-Z0-9]{8,20})',
                r'(?:BOOKING|BKG)\s*(?:NO|NUMBER|#):?\s*([A-Z0-9]{8,20})'
            ],
            'container_number': [
                r'(?:CONTAINER|CONTENEUR)[\s\w]*:?\s*([A-Z]{4}[0-9]{7})',
                r'(?:CNTR|CTR)\s*(?:NO|NUMBER|#):?\s*([A-Z]{4}[0-9]{7})'
            ],
            'vessel_name': [
                r'(?:VESSEL|NAVIRE|SHIP)[\s\w]*:?\s*([A-Z\s]{3,30})',
                r'(?:VESSEL|NAVIRE)\s*(?:NAME|NOM):?\s*([A-Z\s]{3,30})'
            ],
            'voyage_number': [
                r'(?:VOYAGE|VOY)[\s\w]*:?\s*([A-Z0-9]{3,15})',
                r'(?:VOYAGE|VOY)\s*(?:NO|NUMBER|#):?\s*([A-Z0-9]{3,15})'
            ],
            'port_of_loading': [
                r'(?:PORT OF LOADING|POL|PORT DE CHARGEMENT)[\s\w]*:?\s*([A-Z\s,]{5,40})',
                r'(?:LOADED ON BOARD|CHARGÉ À BORD)[\s\w]*:?\s*([A-Z\s,]{5,40})'
            ],
            'port_of_discharge': [
                r'(?:PORT OF DISCHARGE|POD|PORT DE DÉCHARGEMENT)[\s\w]*:?\s*([A-Z\s,]{5,40})',
                r'(?:DISCHARGE|DÉCHARGEMENT)[\s\w]*:?\s*([A-Z\s,]{5,40})'
            ],
            'shipper': [
                r'(?:SHIPPER|EXPÉDITEUR|CHARGEUR)[\s\w]*:?\s*([A-Za-z\s\d,.\-]{10,100})',
                r'(?:SHIPPER|EXPÉDITEUR)\s*:?\s*([A-Za-z\s\d,.\-]{10,100})'
            ],
            'consignee': [
                r'(?:CONSIGNEE|DESTINATAIRE|RÉCEPTIONNAIRE)[\s\w]*:?\s*([A-Za-z\s\d,.\-]{10,100})',
                r'(?:CONSIGNEE|DESTINATAIRE)\s*:?\s*([A-Za-z\s\d,.\-]{10,100})'
            ],
            'notify_party': [
                r'(?:NOTIFY|NOTIFIER|PARTIE À NOTIFIER)[\s\w]*:?\s*([A-Za-z\s\d,.\-]{10,100})',
                r'(?:NOTIFY PARTY|PARTIE À NOTIFIER)\s*:?\s*([A-Za-z\s\d,.\-]{10,100})'
            ],
            'freight_terms': [
                r'(?:FREIGHT|FRET|PAYABLE)[\s\w]*:?\s*(PREPAID|COLLECT|PAYABLE|PRÉPAYÉ)',
                r'(?:FREIGHT PAYABLE|FRET PAYABLE)\s*:?\s*(PREPAID|COLLECT|PAYABLE|PRÉPAYÉ)'
            ],
            'issue_date': [
                r'(?:ISSUE|ÉMISSION|DATE)[\s\w]*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
                r'(?:ISSUED|ÉMIS)\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
            ],
            'weight': [
                r'(?:WEIGHT|POIDS)[\s\w]*:?\s*(\d+(?:\.\d+)?)\s*(?:KG|LB|MT|T)',
                r'(?:GROSS|BRUT)\s*(?:WEIGHT|POIDS)\s*:?\s*(\d+(?:\.\d+)?)\s*(?:KG|LB|MT|T)'
            ],
            'volume': [
                r'(?:VOLUME|CBM|M3)[\s\w]*:?\s*(\d+(?:\.\d+)?)\s*(?:CBM|M3|M³)',
                r'(?:MEASUREMENT|MESURE)\s*:?\s*(\d+(?:\.\d+)?)\s*(?:CBM|M3|M³)'
            ]
        }
    
    async def parse(self, text: str) -> BillOfLadingData:
        """
        Parse le texte extrait et retourne les données structurées
        
        Args:
            text: Texte extrait par OCR
            
        Returns:
            BillOfLadingData: Données structurées
        """
        try:
            logger.info("Début du parsing du texte extrait")
            
            # Nettoyer le texte
            cleaned_text = self._clean_text(text)
            
            # Extraire les données
            data = BillOfLadingData()
            
            # Numéros de référence
            data.bl_number = self._extract_field(cleaned_text, 'bl_number')
            data.booking_number = self._extract_field(cleaned_text, 'booking_number')
            
            # Parties
            data.shipper = self._extract_party(cleaned_text, 'shipper')
            data.consignee = self._extract_party(cleaned_text, 'consignee')
            data.notify_party = self._extract_party(cleaned_text, 'notify_party')
            
            # Ports
            data.port_of_loading = self._extract_port(cleaned_text, 'port_of_loading')
            data.port_of_discharge = self._extract_port(cleaned_text, 'port_of_discharge')
            
            # Détails de transport
            data.transport_details = self._extract_transport_details(cleaned_text)
            
            # Marchandises
            data.cargo = self._extract_cargo(cleaned_text)
            
            # Conteneurs
            data.containers = self._extract_containers(cleaned_text)
            
            # Conditions
            data.freight_terms = self._extract_field(cleaned_text, 'freight_terms')
            data.issue_date = self._extract_field(cleaned_text, 'issue_date')
            
            # Calculer la confiance
            data.extraction_confidence = self._calculate_confidence(data)
            
            logger.info(f"Parsing terminé avec confiance: {data.extraction_confidence}")
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors du parsing: {str(e)}")
            # Retourner un objet vide en cas d'erreur
            return BillOfLadingData(
                extraction_confidence=0.0,
                raw_text=text
            )
    
    def _clean_text(self, text: str) -> str:
        """Nettoie le texte extrait"""
        # Supprimer les caractères spéciaux
        cleaned = re.sub(r'[^\w\s\-\.:,/()#]', ' ', text)
        
        # Normaliser les espaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Convertir en majuscules pour faciliter la recherche
        cleaned = cleaned.upper()
        
        return cleaned.strip()
    
    def _extract_field(self, text: str, field_name: str) -> Optional[str]:
        """Extrait un champ spécifique du texte"""
        if field_name not in self.patterns:
            return None
        
        for pattern in self.patterns[field_name]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_party(self, text: str, party_type: str) -> Optional[Party]:
        """Extrait les informations d'une partie"""
        party_info = self._extract_field(text, party_type)
        
        if not party_info:
            return None
        
        # Essayer de parser l'adresse
        lines = party_info.split('\n')
        
        party = Party()
        if lines:
            party.name = lines[0].strip()
            if len(lines) > 1:
                party.address = '\n'.join(lines[1:]).strip()
        
        return party
    
    def _extract_port(self, text: str, port_type: str) -> Optional[Port]:
        """Extrait les informations d'un port"""
        port_info = self._extract_field(text, port_type)
        
        if not port_info:
            return None
        
        port = Port()
        port.name = port_info.strip()
        
        return port
    
    def _extract_transport_details(self, text: str) -> Optional[TransportDetails]:
        """Extrait les détails de transport"""
        transport = TransportDetails()
        
        transport.vessel_name = self._extract_field(text, 'vessel_name')
        transport.voyage_number = self._extract_field(text, 'voyage_number')
        transport.bl_number = self._extract_field(text, 'bl_number')
        transport.booking_number = self._extract_field(text, 'booking_number')
        
        # Retourner None si aucune information trouvée
        if not any([transport.vessel_name, transport.voyage_number, transport.bl_number]):
            return None
        
        return transport
    
    def _extract_cargo(self, text: str) -> List[Cargo]:
        """Extrait les informations de marchandises"""
        cargo_list = []
        
        # Pattern pour détecter les descriptions de marchandises
        cargo_patterns = [
            r'(?:DESCRIPTION|GOODS|MARCHANDISES)[\s\w]*:?\s*([A-Za-z\s\d,.\-]{10,200})',
            r'(?:COMMODITY|PRODUIT)[\s\w]*:?\s*([A-Za-z\s\d,.\-]{10,200})'
        ]
        
        for pattern in cargo_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cargo = Cargo()
                cargo.description = match.strip()
                cargo.weight = self._extract_field(text, 'weight')
                cargo.volume = self._extract_field(text, 'volume')
                cargo_list.append(cargo)
                break  # Prendre seulement la première correspondance
        
        return cargo_list
    
    def _extract_containers(self, text: str) -> List[Container]:
        """Extrait les informations des conteneurs"""
        containers = []
        
        container_numbers = []
        for pattern in self.patterns['container_number']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            container_numbers.extend(matches)
        
        for number in container_numbers:
            container = Container()
            container.number = number.strip()
            containers.append(container)
        
        return containers
    
    def _calculate_confidence(self, data: BillOfLadingData) -> float:
        """Calcule la confiance de l'extraction"""
        total_fields = 0
        filled_fields = 0
        
        # Compter les champs principaux
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
        
        # Bonus pour les champs additionnels
        bonus_fields = [
            data.booking_number,
            data.transport_details,
            data.cargo,
            data.containers
        ]
        
        for field in bonus_fields:
            total_fields += 0.5
            if field:
                filled_fields += 0.5
        
        if total_fields == 0:
            return 0.0
        
        confidence = filled_fields / total_fields
        return min(confidence, 1.0)