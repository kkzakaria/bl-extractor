from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class Party(BaseModel):
    """Représente une partie impliquée dans le connaissement"""
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    contact: Optional[str] = None

class Port(BaseModel):
    """Représente un port"""
    name: Optional[str] = None
    code: Optional[str] = None
    country: Optional[str] = None

class Cargo(BaseModel):
    """Représente une marchandise"""
    description: Optional[str] = None
    quantity: Optional[str] = None
    weight: Optional[str] = None
    volume: Optional[str] = None
    package_type: Optional[str] = None
    marks_numbers: Optional[str] = None

class Container(BaseModel):
    """Représente un conteneur"""
    number: Optional[str] = None
    size: Optional[str] = None
    type: Optional[str] = None
    seal_number: Optional[str] = None

class TransportDetails(BaseModel):
    """Détails du transport"""
    vessel_name: Optional[str] = None
    voyage_number: Optional[str] = None
    booking_number: Optional[str] = None
    bl_number: Optional[str] = None
    departure_date: Optional[str] = None
    arrival_date: Optional[str] = None

class BillOfLadingData(BaseModel):
    """Modèle principal pour les données du connaissement"""
    
    # Numéros de référence
    bl_number: Optional[str] = Field(None, description="Numéro du connaissement")
    booking_number: Optional[str] = Field(None, description="Numéro de réservation")
    
    # Parties impliquées
    shipper: Optional[Party] = Field(None, description="Expéditeur")
    consignee: Optional[Party] = Field(None, description="Destinataire")
    notify_party: Optional[Party] = Field(None, description="Partie à notifier")
    carrier: Optional[Party] = Field(None, description="Transporteur")
    
    # Ports
    port_of_loading: Optional[Port] = Field(None, description="Port de chargement")
    port_of_discharge: Optional[Port] = Field(None, description="Port de déchargement")
    port_of_delivery: Optional[Port] = Field(None, description="Port de livraison")
    
    # Transport
    transport_details: Optional[TransportDetails] = Field(None, description="Détails du transport")
    
    # Marchandises
    cargo: List[Cargo] = Field(default_factory=list, description="Liste des marchandises")
    containers: List[Container] = Field(default_factory=list, description="Liste des conteneurs")
    
    # Conditions
    freight_terms: Optional[str] = Field(None, description="Conditions de fret")
    payment_terms: Optional[str] = Field(None, description="Conditions de paiement")
    delivery_terms: Optional[str] = Field(None, description="Conditions de livraison")
    
    # Dates
    issue_date: Optional[str] = Field(None, description="Date d'émission")
    
    # Métadonnées d'extraction
    extraction_confidence: Optional[float] = Field(None, description="Confiance de l'extraction (0-1)")
    extraction_method: Optional[str] = Field(None, description="Méthode d'extraction utilisée")
    raw_text: Optional[str] = Field(None, description="Texte brut extrait")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }