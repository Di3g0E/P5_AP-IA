"""
Módulo de notificaciones dual (WhatsApp y Telegram) para el sistema biométrico.
"""

import os
import time
import requests
from typing import Optional, Any
from loguru import logger

try:
    import pywhatkit
    PYWHATKIT_AVAILABLE = True
except ImportError:
    PYWHATKIT_AVAILABLE = False
    logger.warning("pywhatkit no está instalado")


class WhatsAppService:
    """Gestor de notificaciones usando pywhatkit (WhatsApp Web)."""
    
    def __init__(self, phone_number: str):
        if not PYWHATKIT_AVAILABLE:
            raise ImportError("pywhatkit es requerido para las notificaciones de WhatsApp")
        self.phone_number = phone_number
        
    def send_message(self, message: str, wait_time: int = 15) -> bool:
        """Envía mensaje WhatsApp abriendo el navegador."""
        try:
            pywhatkit.sendwhatmsg_instantly(
                phone_no=self.phone_number,
                message=message,
                wait_time=wait_time,
                tab_close=True,
                close_time=3
            )
            logger.info(f"Notificación WhatsApp enviada a {self.phone_number}")
            return True
        except Exception as e:
            logger.error(f"Error enviando WhatsApp: {e}")
            return False


class TelegramService:
    """Gestor de notificaciones usando Telegram Bot API (Invisible/Bot)."""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}/sendMessage"
        
    def send_message(self, message: str) -> bool:
        """Envía mensaje a Telegram de forma invisible (HTTP request)."""
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Notificación Telegram enviada al chat {self.chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error enviando Telegram: {e}")
            return False


def get_notification_message(user_id: str, action: str, success: bool = True, **kwargs) -> str:
    """Genera el texto del mensaje con formato enriquecido."""
    timestamp = time.strftime('%H:%M:%S')
    
    if action == "login":
        status = kwargs.get("status", "SUCCESS" if success else "FAILED")
        msg_text = kwargs.get("message", "Acceso concedido" if success else "Acceso denegado")
        liveness = kwargs.get("liveness")
        similarity = kwargs.get("similarity")
        
        emoji = "✅" if success else "❌"
        lines = [
            f"{emoji} *Login {status}*",
            f"👤 *Usuario*: {user_id}",
            f"📝 *Mensaje*: {msg_text}",
            f"⏰ *Hora*: {timestamp}"
        ]
        return "\n".join(lines)
        
    elif action == "register":
        return f"👤 *Nuevo usuario registrado*\n🆔 ID: {user_id}\n⏰ Hora: {timestamp}"
        
    return f"Notificación de sistema: {action} para {user_id}"


def notify_all(user_id: str, action: str, success: bool = True, **kwargs) -> None:
    """Envía notificaciones por todos los medios configurados."""
    message = get_notification_message(user_id, action, success, **kwargs)
    
    # 1. Intentar Telegram (Bot invisible) - Soporta Markdown
    t_token = os.getenv("TELEGRAM_TOKEN")
    t_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if t_token and t_chat_id:
        tg = TelegramService(t_token, t_chat_id)
        tg.send_message(message)
    
    # 2. Intentar WhatsApp (Navegador) - Texto plano sin asteriscos
    w_phone = os.getenv("PYWHATKIT_PHONE")
    if w_phone and PYWHATKIT_AVAILABLE:
        ws = WhatsAppService(w_phone)
        plain_message = message.replace("*", "")
        ws.send_message(plain_message)


# Funciones de compatibilidad con el código anterior
def notify_login_success(user_id: str, success: bool = True, **kwargs) -> None:
    notify_all(user_id, "login", success, **kwargs)

def notify_user_registered(user_id: str) -> None:
    notify_all(user_id, "register")

