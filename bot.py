import os
import requests
import time

# ==============================
# CONFIGURACIÓN
# ==============================

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ==============================
# VERIFICAR CONFIGURACIÓN
# ==============================

def validar_configuracion():
    faltantes = []

    if not TWELVE_DATA_API_KEY:
        faltantes.append("TWELVE_DATA_API_KEY")

    if not TELEGRAM_BOT_TOKEN:
        faltantes.append("TELEGRAM_BOT_TOKEN")

    if not TELEGRAM_CHAT_ID:
        faltantes.append("TELEGRAM_CHAT_ID")

    if faltantes:
        print("Faltan estas variables:")
        for variable in faltantes:
            print(variable)
        return False

    print("Configuración correcta.")
    return True


# ==============================
# ENVIAR MENSAJE A TELEGRAM
# ==============================

def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    datos = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje
    }

    respuesta = requests.post(url, data=datos, timeout=20)

    if respuesta.status_code == 200:
        print("Mensaje enviado a Telegram
