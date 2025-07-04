import logging
from typing import Optional, Dict, Any, List
import tempfile
import os
import json

logger = logging.getLogger(__name__)

class DoclingProcessor:
    """Processeur Docling pour l'extraction de documents structurés"""
    
    def __init__(self):
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Vérifie si Docling est disponible"""
        try:
            import docling
            return True
        except ImportError:
            logger.warning("Docling non disponible. Installation: pip install docling")
            return False
    
    async def extract_text(self, file_path: str) -> str:
        """
        Extrait le texte structuré avec Docling
        
        Args:
            file_path: Chemin vers le fichier PDF
            
        Returns:
            str: Texte extrait avec structure préservée
        """
        if not self.available:
            raise Exception("Docling non disponible")
        
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            
            # Créer le convertisseur
            converter = DocumentConverter()
            
            # Extraire le document
            result = converter.convert(file_path)
            
            # Obtenir le texte structuré
            structured_text = result.document.export_to_text()
            
            logger.info(f"Extraction Docling réussie: {len(structured_text)} caractères")
            
            return structured_text
            
        except Exception as e:
            logger.error(f"Erreur Docling: {str(e)}")
            raise Exception(f"Erreur Docling: {str(e)}")
    
    async def extract_structured_data(self, file_path: str) -> Dict[str, Any]:
        """
        Extrait les données structurées directement
        
        Args:
            file_path: Chemin vers le fichier PDF
            
        Returns:
            Dict: Données structurées extraites
        """
        if not self.available:
            raise Exception("Docling non disponible")
        
        try:
            from docling.document_converter import DocumentConverter
            
            converter = DocumentConverter()
            result = converter.convert(file_path)
            
            # Obtenir la structure du document
            document = result.document
            
            # Extraire les éléments structurés
            structured_data = {
                "title": self._extract_title(document),
                "tables": self._extract_tables(document),
                "text_blocks": self._extract_text_blocks(document),
                "metadata": self._extract_metadata(document),
                "bill_of_lading_sections": self._extract_bl_sections(document)
            }
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Erreur extraction structurée Docling: {str(e)}")
            raise Exception(f"Erreur extraction structurée Docling: {str(e)}")
    
    def _extract_bl_sections(self, document) -> Dict[str, Any]:
        """Extrait les sections spécifiques aux connaissements"""
        try:
            bl_sections = {
                "header_info": [],
                "parties": [],
                "ports": [],
                "cargo_info": [],
                "transport_details": [],
                "footer_info": []
            }
            
            # Analyser les blocs de texte pour identifier les sections
            text_blocks = self._extract_text_blocks(document)
            
            for block in text_blocks:
                text = block.get("text", "").upper()
                
                # Identification des sections par mots-clés
                if any(keyword in text for keyword in ["BILL OF LADING", "B/L", "CONNAISSEMENT"]):
                    bl_sections["header_info"].append(block)
                elif any(keyword in text for keyword in ["SHIPPER", "EXPÉDITEUR", "CONSIGNEE", "DESTINATAIRE"]):
                    bl_sections["parties"].append(block)
                elif any(keyword in text for keyword in ["PORT OF LOADING", "PORT OF DISCHARGE", "PORT"]):
                    bl_sections["ports"].append(block)
                elif any(keyword in text for keyword in ["VESSEL", "VOYAGE", "NAVIRE"]):
                    bl_sections["transport_details"].append(block)
                elif any(keyword in text for keyword in ["DESCRIPTION", "QUANTITY", "WEIGHT", "MARCHANDISES"]):
                    bl_sections["cargo_info"].append(block)
                else:
                    bl_sections["footer_info"].append(block)
            
            return bl_sections
            
        except Exception as e:
            logger.error(f"Erreur extraction sections BL: {str(e)}")
            return {}
    
    def _extract_title(self, document) -> Optional[str]:
        """Extrait le titre du document"""
        try:
            # Rechercher les éléments de titre
            for element in document.texts:
                if hasattr(element, 'label') and 'title' in element.label.lower():
                    return element.text
            return None
        except:
            return None
    
    def _extract_tables(self, document) -> list:
        """Extrait les tableaux du document"""
        try:
            tables = []
            for table in document.tables:
                table_data = {
                    "data": table.export_to_dataframe().to_dict() if hasattr(table, 'export_to_dataframe') else None,
                    "bbox": table.prov[0].bbox if table.prov else None
                }
                tables.append(table_data)
            return tables
        except:
            return []
    
    def _extract_text_blocks(self, document) -> list:
        """Extrait les blocs de texte avec leur position"""
        try:
            text_blocks = []
            for text in document.texts:
                block = {
                    "text": text.text,
                    "bbox": text.prov[0].bbox if text.prov else None,
                    "label": text.label if hasattr(text, 'label') else None
                }
                text_blocks.append(block)
            return text_blocks
        except:
            return []
    
    def _extract_metadata(self, document) -> Dict[str, Any]:
        """Extrait les métadonnées du document"""
        try:
            return {
                "page_count": len(document.pages) if hasattr(document, 'pages') else None,
                "text_length": len(document.export_to_text()),
                "has_tables": len(document.tables) > 0 if hasattr(document, 'tables') else False,
                "has_images": len(document.pictures) > 0 if hasattr(document, 'pictures') else False
            }
        except:
            return {}
    
    def is_available(self) -> bool:
        """Retourne True si Docling est disponible"""
        return self.available