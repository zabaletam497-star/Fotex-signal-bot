import os
import requests
from datetime import datetime

# ==============================
# CONFIGURACIÓN
# ==============================

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL = "EUR/USD"
INTERVAL = "5min"


# ==============================
# VALIDAR CONFIGURACIÓN
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

    return True


# ==============================
# OBTENER DATOS DE TWELVE DATA
# ==============================

def obtener_datos():
    url = "https://api.twelvedata.com/time_series"

    parametros = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": 100,
        "apikey": TWELVE_DATA_API_KEY
    }

    respuesta = requests.get(url,
