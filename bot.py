import os
import requests


TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def verificar_variables():
    faltantes = []

    if not TWELVE_DATA_API_KEY:
        faltantes.append("TWELVE_DATA_API_KEY")

    if not TELEGRAM_BOT_TOKEN:
        faltantes.append("TELEGRAM_BOT_TOKEN")

    if not TELEGRAM_CHAT_ID:
        faltantes.append("TELEGRAM_CHAT_ID")

    if faltantes:
        print("Faltan estas variables en GitHub Secrets:")
        for variable in faltantes:
            print(f"- {variable}")
        return False

    return True


def obtener_datos():
    url = "https://api.twelvedata.com/time_series"

    parametros = {
        "symbol": "EUR/USD",
        "interval": "5min",
        "outputsize": 1,
        "apikey": TWELVE_DATA_API_KEY
    }

    respuesta = requests.get(url, params=parametros, timeout=30)
    datos = respuesta.json()

    if "status" in datos and datos["status"] == "error":
        print("Error de Twelve Data:")
        print(datos)
        return None

    if "values" not in datos:
        print("Twelve Data no devolvió datos:")
        print(datos)
        return None

    return datos["values"][0]


def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    datos = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje
    }

    respuesta = requests.post(url, data=datos, timeout=30)

    if respuesta.ok:
        print("Mensaje enviado a Telegram correctamente.")
        return True

    print("Error al enviar el mensaje a Telegram:")
    print(respuesta.text)
    return False


def main():
    print("Iniciando Bot Forex...")

    if not verificar_variables():
        raise SystemExit(1)

    print("Variables de configuración encontradas.")

    vela = obtener_datos()

    if vela is None:
        print("No se pudieron obtener los datos de Twelve Data.")
        raise SystemExit(1)

    mensaje = (
        "Bot Forex funcionando correctamente.\n\n"
        "Par: EUR/USD\n"
        "Intervalo: 5 minutos\n\n"
        f"Fecha: {vela.get('datetime', 'N/D')}\n"
        f"Apertura: {vela.get('open', 'N/D')}\n"
        f"Máximo: {vela.get('high', 'N/D')}\n"
        f"Mínimo: {vela.get('low', 'N/D')}\n"
        f"Cierre: {vela.get('close', 'N/D')}"
    )

    if not enviar_telegram(mensaje):
        raise SystemExit(1)

    print("Bot Forex finalizado correctamente.")


if __name__ == "__main__":
    main()
