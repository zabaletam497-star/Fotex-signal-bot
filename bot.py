"""
BOT DE SEÑALES FOREX
M5 + M15
13 FILTROS
REQUISITO DE SEÑAL: 11/13
Twelve Data + Telegram
SOLO GENERA SEÑALES. NO EJECUTA OPERACIONES.

Variables de entorno necesarias:


import os

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
Opcionales:

SCAN_INTERVAL_SECONDS=60
MIN_FILTERS=11
COOLDOWN_MINUTES=30
MAX_SYMBOLS_PER_SCAN=20
ACTIVE_HOURS_START=07:00
ACTIVE_HOURS_END=19:00
TIMEZONE=America/Bogota
"""

import os
import time
import logging
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURACIÓN
# ============================================================

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TIMEZONE_NAME = os.getenv("TIMEZONE", "America/Bogota")

SCAN_INTERVAL_SECONDS = int(
    os.getenv("SCAN_INTERVAL_SECONDS", "60")
)

MIN_FILTERS = int(
    os.getenv("MIN_FILTERS", "11")
)

COOLDOWN_MINUTES = int(
    os.getenv("COOLDOWN_MINUTES", "30")
)

MAX_SYMBOLS_PER_SCAN = int(
    os.getenv("MAX_SYMBOLS_PER_SCAN", "20")
)

ACTIVE_HOURS_START = os.getenv(
    "ACTIVE_HOURS_START",
    "07:00"
)

ACTIVE_HOURS_END = os.getenv(
    "ACTIVE_HOURS_END",
    "19:00"
)

TIMEZONE = ZoneInfo(TIMEZONE_NAME)

API_URL = "https://api.twelvedata.com/time_series"


# ============================================================
# UNIVERSO FOREX
# ============================================================

FOREX_PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD",

    "EUR/GBP",
    "EUR/JPY",
    "EUR/CHF",
    "EUR/AUD",
    "EUR/CAD",
    "EUR/NZD",

    "GBP/JPY",
    "GBP/CHF",
    "GBP/AUD",
    "GBP/CAD",
    "GBP/NZD",

    "AUD/JPY",
    "AUD/NZD",
    "AUD/CAD",
    "AUD/CHF",

    "CAD/JPY",
    "CHF/JPY",

    "NZD/JPY",
    "NZD/CAD",
    "NZD/CHF",
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("FOREX_SIGNAL_BOT")


# ============================================================
# CONTROL DE SEÑALES
# ============================================================

last_signal_time = {}


# ============================================================
# VALIDACIÓN DE CONFIGURACIÓN
# ============================================================

def validate_configuration():
    missing = []

    if not TWELVE_DATA_API_KEY:
        missing.append("TWELVE_DATA_API_KEY")

    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        raise RuntimeError(
            "Faltan variables de entorno: "
            + ", ".join(missing)
        )


# ============================================================
# HORA ACTUAL
# ============================================================

def now_local():
    return datetime.now(TIMEZONE)


# ============================================================
# HORARIO ACTIVO
# ============================================================

def is_active_market_window():
    """
    El proceso puede permanecer encendido 24/7.
    El análisis se concentra en el horario configurado.
    """

    current = now_local()

    start_hour, start_minute = map(
        int,
        ACTIVE_HOURS_START.split(":")
    )

    end_hour, end_minute = map(
        int,
        ACTIVE_HOURS_END.split(":")
    )

    start_minutes = start_hour * 60 + start_minute
    end_minutes = end_hour * 60 + end_minute

    current_minutes = (
        current.hour * 60 +
        current.minute
    )

    if start_minutes <= end_minutes:
        return (
            start_minutes
            <= current_minutes
            <= end_minutes
        )

    return (
        current_minutes >= start_minutes
        or current_minutes <= end_minutes
    )


# ============================================================
# TWELVE DATA
# ============================================================

def get_market_data(symbol, interval, outputsize=250):
    """
    Descarga velas reales desde Twelve Data.
    """

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "timezone": "UTC",
        "order": "asc",
    }

    try:
        response = requests.get(
            API_URL,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if data.get("status") == "error":
            logger.warning(
                "Twelve Data error %s: %s",
                symbol,
                data.get("message")
            )
            return None

        values = data.get("values")

        if not values:
            return None

        df = pd.DataFrame(values)

        required = [
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]

        for column in required:
            if column not in df.columns:
                return None

        for column in [
            "open",
            "high",
            "low",
            "close"
        ]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(
                df["volume"],
                errors="coerce"
            )

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close"
            ]
        )

        df = df.sort_values("datetime")
        df = df.reset_index(drop=True)

        return df

    except Exception as error:
        logger.error(
            "Error descargando %s %s: %s",
            symbol,
            interval,
            error
        )
        return None


# ============================================================
# INDICADORES
# ============================================================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def sma(series, period):
    return series.rolling(
        period
    ).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    result = 100 - (
        100 / (1 + rs)
    )

    return result.fillna(50)


def macd(series):
    fast = ema(series, 12)
    slow = ema(series, 26)

    line = fast - slow
    signal = ema(line, 9)

    histogram = line - signal

    return line, signal, histogram


def bollinger(series, period=20, deviations=2):
    middle = sma(series, period)
    std = series.rolling(period).std()

    upper = middle + deviations * std
    lower = middle - deviations * std

    return upper, middle, lower


def stochastic(df, period=14):
    lowest = df["low"].rolling(period).min()
    highest = df["high"].rolling(period).max()

    denominator = (
        highest - lowest
    ).replace(0, np.nan)

    k = (
        100 *
        (df["close"] - lowest)
        / denominator
    )

    d = k.rolling(3).mean()

    return (
        k.fillna(50),
        d.fillna(50)
    )


def atr(df, period=14):
    previous_close = df["close"].shift(1)

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.rolling(
        period
    ).mean()


def adx(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move)
        & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move)
        & (down_move > 0),
        down_move,
        0
    )

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr_value = tr.rolling(period).mean()

    plus_di = (
        100 *
        pd.Series(plus_dm, index=df.index)
        .rolling(period)
        .mean()
        / atr_value
    )

    minus_di = (
        100 *
        pd.Series(minus_dm, index=df.index)
        .rolling(period)
        .mean()
        / atr_value
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di).replace(
            0,
            np.nan
        )
    )

    return dx.rolling(period).mean()


def cci(df, period=20):
    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    mean = typical_price.rolling(
        period
    ).mean()

    mean_deviation = (
        typical_price
        .rolling(period)
        .apply(
            lambda x: np.mean(
                np.abs(x - np.mean(x))
            ),
            raw=True
        )
    )

    result = (
        typical_price - mean
    ) / (
        0.015 *
        mean_deviation.replace(
            0,
            np.nan
        )
    )

    return result.fillna(0)


# ============================================================
# PREPARAR INDICADORES
# ============================================================

def prepare_indicators(df):
    df = df.copy()

    df["ema9"] = ema(
        df["close"],
        9
    )

    df["ema21"] = ema(
        df["close"],
        21
    )

    df["ema50"] = ema(
        df["close"],
        50
    )

    df["ema200"] = ema(
        df["close"],
        200
    )

    df["rsi"] = rsi(
        df["close"],
        14
    )

    (
        df["macd"],
        df["macd_signal"],
        df["macd_hist"]
    ) = macd(
        df["close"]
    )

    (
        df["bb_upper"],
        df["bb_middle"],
        df["bb_lower"]
    ) = bollinger(
        df["close"]
    )

    (
        df["stoch_k"],
        df["stoch_d"]
    ) = stochastic(df)

    df["adx"] = adx(df)

    df["cci"] = cci(df)

    df["atr"] = atr(df)

    df["sma20"] = sma(
        df["close"],
        20
    )

    df["sma50"] = sma(
        df["close"],
        50
    )

    return df


# ============================================================
# 13 FILTROS
# ============================================================

def filter_1_ema_cross(df):
    """
    EMA 9 / EMA 21
    """

    row = df.iloc[-1]

    if row["ema9"] > row["ema21"]:
        return "BUY"

    if row["ema9"] < row["ema21"]:
        return "SELL"

    return "NO CONFIRMA"


def filter_2_trend(df):
    """
    Tendencia EMA 50 / EMA 200
    """

    row = df.iloc[-1]

    if row["ema50"] > row["ema200"]:
        return "BUY"

    if row["ema50"] < row["ema200"]:
        return "SELL"

    return "NO CONFIRMA"


def filter_3_rsi(df):
    row = df.iloc[-1]

    if 50 <= row["rsi"] <= 70:
        return "BUY"

    if 30 <= row["rsi"] < 50:
        return "SELL"

    return "NO CONFIRMA"


def filter_4_macd(df):
    row = df.iloc[-1]

    if (
        row["macd"] > row["macd_signal"]
        and row["macd_hist"] > 0
    ):
        return "BUY"

    if (
        row["macd"] < row["macd_signal"]
        and row["macd_hist"] < 0
    ):
        return "SELL"

    return "NO CONFIRMA"


def filter_5_bollinger(df):
    row = df.iloc[-1]

    if row["close"] > row["bb_middle"]:
        return "BUY"

    if row["close"] < row["bb_middle"]:
        return "SELL"

    return "NO CONFIRMA"


def filter_6_stochastic(df):
    row = df.iloc[-1]

    if (
        row["stoch_k"] > row["stoch_d"]
        and row["stoch_k"] > 50
    ):
        return "BUY"

    if (
        row["stoch_k"] < row["stoch_d"]
        and row["stoch_k"] < 50
    ):
        return "SELL"

    return "NO CONFIRMA"


def filter_7_adx(df):
    row = df.iloc[-1]

    if pd.isna(row["adx"]):
        return "NO CONFIRMA"

    if row["adx"] >= 25:
        if row["ema9"] > row["ema21"]:
            return "BUY"

        if row["ema9"] < row["ema21"]:
            return "SELL"

    return "NO CONFIRMA"


def filter_8_cci(df):
    row = df.iloc[-1]

    if row["cci"] > 0:
        return "BUY"

    if row["cci"] < 0:
        return "SELL"

    return "NO CONFIRMA"


def filter_9_price_structure(df):
    """
    Estructura simple de máximos/mínimos recientes.
    """

    if len(df) < 10:
        return "NO CONFIRMA"

    recent = df.iloc[-5:]

    previous = df.iloc[-10:-5]

    recent_high = recent["high"].max()
    previous_high = previous["high"].max()

    recent_low = recent["low"].min()
    previous_low = previous["low"].min()

    if (
        recent_high > previous_high
        and recent_low > previous_low
    ):
        return "BUY"

    if (
        recent_high < previous_high
        and recent_low < previous_low
    ):
        return "SELL"

    return "NO CONFIRMA"


def filter_10_support_resistance(df):
    """
    Proximidad a máximos/mínimos recientes.
    """

    if len(df) < 30:
        return "NO CONFIRMA"

    row = df.iloc[-1]

    resistance = df["high"].iloc[-30:-1].max()
    support = df["low"].iloc[-30:-1].min()

    price = row["close"]

    distance_resistance = abs(
        resistance - price
    )

    distance_support = abs(
        price - support
    )

    atr_value = row["atr"]

    if pd.isna(atr_value) or atr_value <= 0:
        return "NO CONFIRMA"

    if (
        price > df["close"].iloc[-2]
        and distance_resistance > atr_value * 0.5
    ):
        return "BUY"

    if (
        price < df["close"].iloc[-2]
        and distance_support > atr_value * 0.5
    ):
        return "SELL"

    return "NO CONFIRMA"


def filter_11_volatility(df):
    """
    ATR.
    Evita mercados completamente planos.
    """

    row = df.iloc[-1]

    if pd.isna(row["atr"]):
        return "NO CONFIRMA"

    recent_atr = df["atr"].iloc[-20:].mean()

    if pd.isna(recent_atr):
        return "NO CONFIRMA"

    if row["atr"] > recent_atr * 1.05:
        if row["close"] > row["open"]:
            return "BUY"

        if row["close"] < row["open"]:
            return "SELL"

    return "NO CONFIRMA"


def filter_12_momentum(df):
    """
    Momentum de precio.
    """

    if len(df) < 6:
        return "NO CONFIRMA"

    current = df["close"].iloc[-1]
    previous = df["close"].iloc[-6]

    if current > previous:
        return "BUY"

    if current < previous:
        return "SELL"

    return "NO CONFIRMA"


def filter_13_m15_context(df_m5, df_m15):
    """
    Confirma la dirección del contexto M15.
    """

    if len(df_m15) < 50:
        return "NO CONFIRMA"

    row = df_m15.iloc[-1]

    if row["ema9"] > row["ema21"]:
        return "BUY"

    if row["ema9"] < row["ema21"]:
        return "SELL"

    return "NO CONFIRMA"


# ============================================================
# EJECUTAR LOS 13 FILTROS
# ============================================================

def evaluate_filters(df_m5, df_m15):

    filters = {
        "EMA 9/21": filter_1_ema_cross(df_m5),
        "Tendencia EMA 50/200": filter_2_trend(df_m5),
        "RSI": filter_3_rsi(df_m5),
        "MACD": filter_4_macd(df_m5),
        "Bollinger": filter_5_bollinger(df_m5),
        "Estocástico": filter_6_stochastic(df_m5),
        "ADX": filter_7_adx(df_m5),
        "CCI": filter_8_cci(df_m5),
        "Estructura": filter_9_price_structure(df_m5),
        "Soporte/Resistencia": filter_10_support_resistance(df_m5),
        "Volatilidad ATR": filter_11_volatility(df_m5),
        "Momentum": filter_12_momentum(df_m5),
        "Contexto M15": filter_13_m15_context(
            df_m5,
            df_m15
        ),
    }

    buy_count = sum(
        1
        for value in filters.values()
        if value == "BUY"
    )

    sell_count = sum(
        1
        for value in filters.values()
        if value == "SELL"
    )

    if buy_count >= MIN_FILTERS:
        direction = "BUY"
        score = buy_count

    elif sell_count >= MIN_FILTERS:
        direction = "SELL"
        score = sell_count

    else:
        direction = None
        score = max(
            buy_count,
            sell_count
        )

    return (
        direction,
        score,
        buy_count,
        sell_count,
        filters
    )


# ============================================================
# CALCULAR NIVELES
# ============================================================

def calculate_levels(df, direction):
    """
    Calcula niveles orientativos basados en ATR.

    IMPORTANTE:
    Estos niveles son informativos.
    No ejecutan ninguna operación.
    """

    row = df.iloc[-1]

    price = float(row["close"])
    atr_value = float(row["atr"])

    if not np.isfinite(atr_value) or atr_value <= 0:
        atr_value = price * 0.001

    if direction == "BUY":

        entry = price

        stop_loss = (
            price -
            atr_value * 1.5
        )

        take_profit_1 = (
            price +
            atr_value * 1.5
        )

        take_profit_2 = (
            price +
            atr_value * 2.5
        )

    else:

        entry = price

        stop_loss = (
            price +
            atr_value * 1.5
        )

        take_profit_1 = (
            price -
            atr_value * 1.5
        )

        take_profit_2 = (
            price -
            atr_value * 2.5
        )

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
    }


# ============================================================
# COOLDOWN
# ============================================================

def signal_allowed(symbol, direction):
    key = f"{symbol}:{direction}"

    last_time = last_signal_time.get(key)

    if last_time is None:
        return True

    elapsed = (
        datetime.utcnow() -
        last_time
    ).total_seconds()

    return elapsed >= (
        COOLDOWN_MINUTES * 60
    )


def register_signal(symbol, direction):
    key = f"{symbol}:{direction}"

    last_signal_time[key] = datetime.utcnow()


# ============================================================
# FORMATO DE SEÑAL
# ============================================================

def format_price(value):
    value = float(value)

    if abs(value) >= 100:
        return f"{value:.3f}"

    if abs(value) >= 10:
        return f"{value:.4f}"

    return f"{value:.5f}"


def build_signal_message(
    symbol,
    direction,
    score,
    buy_count,
    sell_count,
    filters,
    levels
):

    current_time = now_local()

    direction_text = (
        "🟢 COMPRA"
        if direction == "BUY"
        else
        "🔴 VENTA"
    )

    lines = []

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("📊 SEÑAL FOREX")
    lines.append("━━━━━━━━━━━━━━━━━━━━")

    lines.append(
        f"💱 PAR: {symbol}"
    )

    lines.append(
        f"🎯 DIRECCIÓN: {direction_text}"
    )

    lines.append(
        "⏱️ TEMPORALIDAD: M5"
    )

    lines.append(
        "🧭 CONTEXTO: M15"
    )

    lines.append(
        f"📈 CONFIRMACIÓN: {score}/13"
    )

    lines.append("")

    lines.append(
        "📌 FILTROS"
    )

    for number, (
        name,
        result
    ) in enumerate(
        filters.items(),
        start=1
    ):

        if result == "BUY":
            icon = "🟢"

        elif result == "SELL":
            icon = "🔴"

        else:
            icon = "⚪"

        lines.append(
            f"{number:02d}. {icon} {name}: {result}"
        )

    lines.append("")

    lines.append("💰 NIVELES ORIENTATIVOS")

    lines.append(
        f"Entrada: {format_price(levels['entry'])}"
    )

    lines.append(
        f"SL: {format_price(levels['stop_loss'])}"
    )

    lines.append(
        f"TP1: {format_price(levels['take_profit_1'])}"
    )

    lines.append(
        f"TP2: {format_price(levels['take_profit_2'])}"
    )

    lines.append("")

    lines.append(
        f"🟢 BUY filtros: {buy_count}/13"
    )

    lines.append(
        f"🔴 SELL filtros: {sell_count}/13"
    )

    lines.append(
        f"🕐 Hora: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    lines.append(
        "📡 Fuente: Twelve Data"
    )

    lines.append("")

    lines.append(
        "⚠️ Señal informativa. "
        "El sistema no ejecuta operaciones."
    )

    lines.append("━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN:
        logger.warning(
            "TELEGRAM_BOT_TOKEN no configurado."
        )
        return False

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("ok"):
            logger.error(
                "Telegram rechazó el mensaje: %s",
                data
            )
            return False

        return True

    except Exception as error:

        logger.error(
            "Error enviando Telegram: %s",
            error
        )

        return False


# ============================================================
# ANALIZAR UN PAR
# ============================================================

def analyze_symbol(symbol):

    logger.info(
        "Analizando %s...",
        symbol
    )

    df_m5 = get_market_data(
        symbol,
        "5min",
        300
    )

    if df_m5 is None:
        return None

    df_m15 = get_market_data(
        symbol,
        "15min",
        300
    )

    if df_m15 is None:
        return None

    if len(df_m5) < 220:
        logger.warning(
            "%s: insuficientes velas M5",
            symbol
        )
        return None

    if len(df_m15) < 220:
        logger.warning(
            "%s: insuficientes velas M15",
            symbol
        )
        return None

    df_m5 = prepare_indicators(
        df_m5
    )

    df_m15 = prepare_indicators(
        df_m15
    )

    (
        direction,
        score,
        buy_count,
        sell_count,
        filters
    ) = evaluate_filters(
        df_m5,
        df_m15
    )

    logger.info(
        "%s | BUY=%s | SELL=%s | SCORE=%s",
        symbol,
        buy_count,
        sell_count,
        score
    )

    if direction is None:
        return None

    if not signal_allowed(
        symbol,
        direction
    ):
        logger.info(
            "%s | señal bloqueada por cooldown",
            symbol
        )
        return None

    levels = calculate_levels(
        df_m5,
        direction
    )

    message = build_signal_message(
        symbol=symbol,
        direction=direction,
        score=score,
        buy_count=buy_count,
        sell_count=sell_count,
        filters=filters,
        levels=levels
    )

    return {
        "symbol": symbol,
        "direction": direction,
        "score": score,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "message": message
    }


# ============================================================
# ESCANER
# ============================================================

def scan_market():

    logger.info(
        "========== INICIO DEL ESCANEO =========="
    )

    symbols = FOREX_PAIRS[
        :MAX_SYMBOLS_PER_SCAN
    ]

    signals_found = 0

    for symbol in symbols:

        try:

            result = analyze_symbol(
                symbol
            )

            if not result:
                continue

            logger.info(
                "SEÑAL ENCONTRADA: %s %s %s/13",
                result["symbol"],
                result["direction"],
                result["score"]
            )

            sent = send_telegram(
                result["message"]
            )

            if sent:

                register_signal(
                    result["symbol"],
                    result["direction"]
                )

                signals_found += 1

                logger.info(
                    "Señal enviada correctamente."
                )

        except Exception as error:

            logger.error(
                "Error analizando %s: %s",
                symbol,
                error
            )

            logger.debug(
                traceback.format_exc()
            )

        time.sleep(1)

    logger.info(
        "========== FIN DEL ESCANEO | "
        "SEÑALES: %s ==========",
        signals_found
    )


# ============================================================
# ESTADO DEL BOT
# ============================================================

def print_status():

    current = now_local()

    logger.info(
        "BOT ACTIVO | %s | Ventana análisis: %s",
        current.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        (
            "ACTIVA"
            if is_active_market_window()
            else "PAUSA"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("       BOT DE SEÑALES FOREX")
    print("==========================================")
    print("Motor: 13 filtros")
    print("Principal: M5")
    print("Contexto: M15")
    print(f"Mínimo señal: {MIN_FILTERS}/13")
    print("Operaciones automáticas: NO")
    print("Telegram: ACTIVO")
    print("Proceso: 24/7")
    print("Análisis concentrado: horario activo")
    print("==========================================")
    print()

    validate_configuration()

    logger.info(
        "Configuración validada correctamente."
    )

    while True:

        try:

            print_status()

            if is_active_market_window():

                scan_market()

            else:

                logger.info(
                    "Fuera de la ventana principal. "
                    "Bot permanece activo."
                )

            time.sleep(
                SCAN_INTERVAL_SECONDS
            )

        except KeyboardInterrupt:

            logger.info(
                "Bot detenido manualmente."
            )

            break

        except Exception as error:

            logger.error(
                "Error principal: %s",
                error
            )

            logger.debug(
                traceback.format_exc()
            )

            time.sleep(30)


# ============================================================
# ARRANQUE
# ============================================================

if __name__ == "__main__":
    main()
