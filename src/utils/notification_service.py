"""
Módulo de notificaciones usando pywhatkit para el sistema biométrico.
"""

import os
import time
from typing import Optional
from loguru import logger

try:
    import pywhatkit
    PYWHATKIT_AVAILABLE = True
except ImportError:
    PYWHATKIT_AVAILABLE = False
    logger.warning("pywhatkit no está instalado")


class NotificationService:
    """Gestor de notificaciones usando pywhatkit (WhatsApp Web)."""
    
    def __init__(self, phone_number: str):
        if not PYWHATKIT_AVAILABLE:
            raise ImportError("pywhatkit es requerido para las notificaciones")
        
        self.phone_number = phone_number
        
    def send_message(self, message: str, wait_time: int = 15) -> bool:
        """
        Envía mensaje WhatsApp usando pywhatkit.
        
        Args:
            message: Mensaje a enviar
            wait_time: Tiempo de espera para cargar WhatsApp Web
            
        Returns:
            True si éxito, False si error
        """
        try:
            # pywhatkit.sendwhatmsg(phone_no, message, time_hour, time_min, wait_time)
            # Usamos la hora actual + 2 minutos para asegurar tiempo suficiente
            import datetime
            now = datetime.datetime.now()
            target_time = now + datetime.timedelta(minutes=2)
            
            pywhatkit.sendwhatmsg(
                phone_no=self.phone_number,
                message=message,
                time_hour=target_time.hour,
                time_min=target_time.minute,
                wait_time=wait_time
            )
            
            logger.info(f"Notificación enviada a {self.phone_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error enviando notificación: {e}")
            return False
    
    def send_login_notification(self, user_id: str, success: bool = True) -> bool:
        """Envía notificación de login."""
        if success:
            message = f"Login exitoso para usuario: {user_id}\nHora: {time.strftime('%H:%M:%S')}"
        else:
            message = f"Falló login para usuario: {user_id}\nHora: {time.strftime('%H:%M:%S')}"
        
        return self.send_message(message)
    
    def send_registration_notification(self, user_id: str) -> bool:
        """Envía notificación de registro."""
        message = f"👤 Nuevo usuario registrado: {user_id}\nHora: {time.strftime('%H:%M:%S')}"
        return self.send_message(message)
    
    @classmethod
    def from_env(cls) -> Optional["NotificationService"]:
        """Crea instancia desde variables de entorno."""
        phone = os.getenv("PYWHATKIT_PHONE")
        
        if not phone:
            logger.warning("Variable PYWHATKIT_PHONE no configurada")
            return None
            
        return cls(phone)


def notify_login_success(user_id: str, success: bool = True) -> None:
    """Notifica login si está configurado pywhatkit."""
    if not PYWHATKIT_AVAILABLE:
        return
        
    notifier = NotificationService.from_env()
    if notifier:
        notifier.send_login_notification(user_id, success)


def notify_user_registered(user_id: str) -> None:
    """Notifica nuevo registro si está configurado pywhatkit."""
    if not PYWHATKIT_AVAILABLE:
        return
        
    notifier = NotificationService.from_env()
    if notifier:
        notifier.send_registration_notification(user_id)
