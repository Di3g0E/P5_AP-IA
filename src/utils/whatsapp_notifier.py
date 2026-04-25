"""
Módulo de notificaciones WhatsApp para el sistema biométrico.
"""

import os
import requests
from typing import Optional
from loguru import logger


class WhatsAppNotifier:
    """Gestor de notificaciones WhatsApp usando API de Meta."""
    
    def __init__(self, token: str, phone_id: str):
        self.token = token
        self.phone_id = phone_id
        self.url = f"https://graph.facebook.com/v21.0/{phone_id}/messages"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def send_notification(self, phone_dest: str, user_name: str, 
                         template_name: str = "hello_world") -> bool:
        """
        Envía notificación WhatsApp.
        
        Args:
            phone_dest: Teléfono destino (formato: 34XXXXXXXXX)
            user_name: Nombre del usuario
            template_name: Plantilla WhatsApp aprobada
            
        Returns:
            True si éxito, False si error
        """
        try:
            data = {
                "messaging_product": "whatsapp",
                "to": phone_dest,
                "type": "template",
                "template": {
                    "name": template_name,
                    "language": {"code": "en_US"}
                }
            }
            
            response = requests.post(self.url, headers=self.headers, json=data)
            
            if response.status_code == 200:
                logger.info(f"Notificación WhatsApp enviada a {user_name}")
                return True
            else:
                logger.error(f"Error WhatsApp API: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error enviando WhatsApp: {e}")
            return False
    
    @classmethod
    def from_env(cls) -> Optional["WhatsAppNotifier"]:
        """Crea instancia desde variables de entorno."""
        token = os.getenv("WHATSAPP_TOKEN")
        phone_id = os.getenv("WHATSAPP_PHONE_ID")
        
        if not token or not phone_id:
            logger.warning("Variables WHATSAPP_TOKEN/WHATSAPP_PHONE_ID no configuradas")
            return None
            
        return cls(token, phone_id)


def notify_login_success(user_id: str, phone: Optional[str] = None) -> None:
    """Notifica login exitoso si está configurado WhatsApp."""
    notifier = WhatsAppNotifier.from_env()
    if notifier and phone:
        notifier.send_notification(phone, user_id)


def notify_user_registered(user_id: str, phone: Optional[str] = None) -> None:
    """Notifica nuevo registro si está configurado WhatsApp."""
    notifier = WhatsAppNotifier.from_env()
    if notifier and phone:
        notifier.send_notification(phone, user_id)
