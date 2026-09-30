import os
import time
import logging
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURACION
# ============================================================

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
MIN_FILTERS = int(os.getenv("MIN_FILTERS", "11"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))
TIMEZONE = os.getenv("TIMEZONE", "America/Bogota")

SYMBOLS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "USD/CAD",
]

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

last_signal_time = {}


# ============================================================
# UTILIDADES
# ============================================================

def now_local():
    return datetime.now(ZoneInfo(TIMEZONE))


def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en GitHub Secrets.")

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
    }

    response = requests.post(TELEGRAM_URL, json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()

    if not data.get("ok"):
        raise RuntimeError(f"Telegram rechazo el mensaje: {data}")

    print("Mensaje enviado a Telegram")


def get_candles(symbol, interval, outputsize=120):
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError("Falta TWELVE_DATA_API_KEY en GitHub Secrets.")

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON",
    }

    response = requests.get(
        TWELVE_DATA_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") == "error":
        message = data.get(
            "message",
            "Error desconocido de Twelve Data.",
        )
        raise RuntimeError(f"Twelve Data: {message}")

    values = data.get("values")

    if not values:
        raise RuntimeError(
            f"Twelve Data no devolvio velas para {symbol} {interval}."
        )

    df = pd.DataFrame(values)

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required:
        if column not in df.columns:
            raise RuntimeError(
                f"Falta la columna {column} en los datos de {symbol}."
            )

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce",
        )
    else:
        df["volume"] = 1.0

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# ============================================================
# INDICADORES
# ============================================================

def add_indicators(df):
    data = df.copy()

    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    data["ema9"] = close.ewm(
        span=9,
        adjust=False,
    ).mean()

    data["ema21"] = close.ewm(
        span=21,
        adjust=False,
    ).mean()

    data["ema50"] = close.ewm(
        span=50,
        adjust=False,
    ).mean()

    data["ema200"] = close.ewm(
        span=200,
        adjust=False,
    ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    data["rsi"] = 100 - (
        100 / (1 + rs)
    )

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    data["macd"] = ema12 - ema26

    data["macd_signal"] = data["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    lowest14 = low.rolling(14).min()
    highest14 = high.rolling(14).max()

    denominator = (
        highest14 - lowest14
    ).replace(
        0,
        np.nan,
    )

    data["stoch_k"] = (
        100 * (close - lowest14) / denominator
    )

    data["stoch_d"] = data["stoch_k"].rolling(3).mean()

    data["bb_mid"] = close.rolling(20).mean()

    bb_std = close.rolling(20).std()

    data["bb_upper"] = (
        data["bb_mid"] + (2 * bb_std)
    )

    data["bb_lower"] = (
        data["bb_mid"] - (2 * bb_std)
    )

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    data["atr"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0),
        0.0,
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        0.0,
    )

    atr14 = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / 14,
            adjust=False,
        ).mean()
        / atr14.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / 14,
            adjust=False,
        ).mean()
        / atr14.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(
            0,
            np.nan,
        )
    )

    data["adx"] = dx.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    data["volume_avg"] = volume.rolling(20).mean()

    return data


# ============================================================
# ANALISIS Y FILTROS
# ============================================================

def analyze_symbol(symbol):
    m5 = add_indicators(
        get_candles(
            symbol,
            "5min",
            220,
        )
    )

    m15 = add_indicators(
        get_candles(
            symbol,
            "15min",
            220,
        )
    )

    if len(m5) < 205 or len(m15) < 205:
        raise RuntimeError(
            f"No hay suficientes velas para analizar {symbol}."
        )

    a = m5.iloc[-1]
    prev = m5.iloc[-2]
    h = m15.iloc[-1]

    buy = 0
    sell = 0

    reasons = []

    # 1. Tendencia principal M15
    if h["ema50"] > h["ema200"]:
        buy += 1
        reasons.append("M15 alcista")
    elif h["ema50"] < h["ema200"]:
        sell += 1
        reasons.append("M15 bajista")

    # 2. Tendencia M5
    if a["ema50"] > a["ema200"]:
        buy += 1
        reasons.append("M5 alcista")
    elif a["ema50"] < a["ema200"]:
        sell += 1
        reasons.append("M5 bajista")

    # 3. EMA 9 / EMA 21
    if a["ema9"] > a["ema21"]:
        buy += 1
        reasons.append("EMA 9/21 alcista")
    elif a["ema9"] < a["ema21"]:
        sell += 1
        reasons.append("EMA 9/21 bajista")

    # 4. RSI
    if 50 < a["rsi"] < 70:
        buy += 1
        reasons.append(
            f"RSI {a['rsi']:.1f} alcista"
        )
    elif 30 < a["rsi"] < 50:
        sell += 1
        reasons.append(
            f"RSI {a['rsi']:.1f} bajista"
        )

    # 5. MACD
    if a["macd"] > a["macd_signal"]:
        buy += 1
        reasons.append("MACD alcista")
    elif a["macd"] < a["macd_signal"]:
        sell += 1
        reasons.append("MACD bajista")

    # 6. Estocastico
    if (
        a["stoch_k"] > a["stoch_d"]
        and a["stoch_k"] < 80
    ):
        buy += 1
        reasons.append("Estocastico alcista")
    elif (
        a["stoch_k"] < a["stoch_d"]
        and a["stoch_k"] > 20
    ):
        sell += 1
        reasons.append("Estocastico bajista")

    # 7. Bandas de Bollinger
    if (
        a["close"] > a["bb_mid"]
        and a["close"] < a["bb_upper"]
    ):
        buy += 1
        reasons.append("Bollinger alcista")
    elif (
        a["close"] < a["bb_mid"]
        and a["close"] > a["bb_lower"]
    ):
        sell += 1
        reasons.append("Bollinger bajista")

    # 8. Volumen
    if a["volume"] >= a["volume_avg"]:
        if a["close"] >= a["open"]:
            buy += 1
            reasons.append(
                "Volumen confirma compra"
            )
        else:
            sell += 1
            reasons.append(
                "Volumen confirma venta"
            )

    # 9. Soporte / resistencia
    support = m5["low"].iloc[-21:-1].min()
    resistance = m5["high"].iloc[-21:-1].max()

    if (
        a["close"] > support
        and a["close"] > prev["close"]
    ):
        buy += 1
        reasons.append("Soporte respetado")
    elif (
        a["close"] < resistance
        and a["close"] < prev["close"]
    ):
        sell += 1
        reasons.append(
            "Resistencia respetada"
        )

    # 10. Fuerza de vela
    body = abs(
        a["close"] - a["open"]
    )

    candle_range = max(
        a["high"] - a["low"],
        1e-12,
    )

    if (
        a["close"] > a["open"]
        and body / candle_range >= 0.5
    ):
        buy += 1
        reasons.append(
            "Vela de fuerza alcista"
        )
    elif (
        a["close"] < a["open"]
        and body / candle_range >= 0.5
    ):
        sell += 1
        reasons.append(
            "Vela de fuerza bajista"
        )

    # 11. ADX
    if a["adx"] >= 20:
        if a["ema9"] > a["ema21"]:
            buy += 1
            reasons.append(
                f"ADX {a['adx']:.1f} confirma compra"
            )
        elif a["ema9"] < a["ema21"]:
            sell += 1
            reasons.append(
                f"ADX {a['adx']:.1f} confirma venta"
            )

    # 12. Sesion activa
    hour = now_local().hour

    if 3 <= hour < 22:
        if a["ema9"] > a["ema21"]:
            buy += 1
            reasons.append("Sesion activa")
        elif a["ema9"] < a["ema21"]:
            sell += 1
            reasons.append("Sesion activa")

    # 13. Volatilidad y calidad de datos
    if pd.notna(a["atr"]) and a["atr"] > 0:
        if a["close"] > a["ema21"]:
            buy += 1
            reasons.append(
                "Volatilidad valida"
            )
        elif a["close"] < a["ema21"]:
            sell += 1
            reasons.append(
                "Volatilidad valida"
            )

    total = max(
        buy,
        sell,
    )

    if (
        buy >= MIN_FILTERS
        and buy > sell
    ):
        direction = "COMPRA"
        confirmations = buy
    elif (
        sell >= MIN_FILTERS
        and sell > buy
    ):
        direction = "VENTA"
        confirmations = sell
    else:
        direction = None
        confirmations = total

    entry = float(a["close"])

    if pd.notna(a["atr"]):
        atr = float(a["atr"])
    else:
        atr = 0.0

    if direction == "COMPRA":
        stop_loss = entry - (1.5 * atr)
        take_profit = entry + (3.0 * atr)
    elif direction == "VENTA":
        stop_loss = entry + (1.5 * atr)
        take_profit = entry - (3.0 * atr)
    else:
        stop_loss = None
        take_profit = None

    return {
        "symbol": symbol,
        "direction": direction,
        "confirmations": confirmations,
        "buy": buy,
        "sell": sell,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "rsi": float(a["rsi"]),
        "adx": float(a["adx"]),
        "time": str(a["datetime"]),
        "reasons": reasons,
    }


# ============================================================
# MENSAJES
# ============================================================

def format_signal(result):
    return (
        "NUEVA SENAL FOREX\n"
        f"Par: {result['symbol']}\n"
        f"Direccion: {result['direction']}\n"
        f"Confirmaciones: {result['confirmations']}/13\n"
        f"Entrada: {result['entry']:.5f}\n"
        f"Stop Loss: {result['stop_loss']:.5f}\n"
        f"Take Profit: {result['take_profit']:.5f}\n"
        f"RSI: {result['rsi']:.2f}\n"
        f"ADX: {result['adx']:.2f}\n"
        f"Hora de vela: {result['time']}\n"
        "MODO DEMO - Solo señal, sin operaciones reales."
    )


def can_send_signal(symbol):
    last_time = last_signal_time.get(symbol)

    if last_time is None:
        return True

    elapsed = (
        time.time() - last_time
    ) / 60

    return elapsed >= COOLDOWN_MINUTES


# ============================================================
# EJECUCION PRINCIPAL
# ============================================================

def main():
    print("Iniciando Bot Forex...")
    print("Python y el codigo fueron cargados correctamente.")

    send_telegram(
        "BOT FOREX INICIADO\n"
        "Conexion con Telegram correcta.\n"
        "Modo DEMO: no ejecuta operaciones reales."
    )

    while True:
        print(
            f"Escaneo iniciado: "
            f"{now_local().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        for symbol in SYMBOLS:
            try:
                result = analyze_symbol(symbol)

                print(
                    f"{symbol} | "
                    f"BUY={result['buy']} | "
                    f"SELL={result['sell']} | "
                    f"senal={result['direction']}"
                )

                if (
                    result["direction"]
                    and can_send_signal(symbol)
                ):
                    send_telegram(
                        format_signal(result)
                    )

                    last_signal_time[symbol] = (
                        time.time()
                    )

            except Exception as error:
                logging.error(
                    "Error analizando %s: %s",
                    symbol,
                    error,
                )

                traceback.print_exc()

        print(
            f"Escaneo terminado. "
            f"Esperando {SCAN_INTERVAL_SECONDS} segundos."
        )

        time.sleep(
            SCAN_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Bot detenido manualmente.")
    except Exception as error:
        logging.error(
            "Error fatal: %s",
            error,
        )

        traceback.print_exc()
        raise
