import openai
import speech_recognition as sr
import requests
import pygame
import tempfile
import io
import json
import vosk
import pyaudio
import signal
import sys
import logging
import asyncio
import keyboard
from tgdbot import initialize_mt5, smc_trading_strategy, get_account_info, mt5, open_position
import os
import threading
import re
from dotenv import load_dotenv
import numpy as np
# Словарь ключевых слов для трейдинга
trading_keywords = {
    'buy': ['купить', 'открой позицию', 'покупка'],
    'sell': ['продать', 'закрой позицию', 'продажа'],
    'gold': ['золото', 'xauusd', 'XAUUSDm'],
    'account': ['баланс', 'счет', 'информация о счете']
}

# Инициализация MetaTrader5
import MetaTrader5 as mt5

# Функция для открытия торговой позиции с проверкой успешности
def open_position(symbol, volume, order_type, reason):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Ошибка: Символ {symbol} не найден.")
        return False

    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid

    if not symbol_info.visible and not mt5.symbol_select(symbol, True):
        logging.error(f"Ошибка: Не удалось активировать торговлю для {symbol}")
        return False

    order = mt5.order_send(
        symbol=symbol,
        action=mt5.TRADE_ACTION_DEAL,
        volume=volume,
        type=order_type,
        price=price,
        deviation=20,
        magic=1234567,
        comment=reason
    )

    if order.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Ошибка при открытии позиции для {symbol}: {order.retcode}. Описание: {mt5.last_error()}")
        return False

    logging.info(f"Позиция {order_type} для {symbol} открыта успешно. Объем: {volume}, Цена: {price}")
    return True

# Логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Инициализация pygame для звуков
pygame.mixer.init()

# Переменные состояния
is_active = False
interrupted = False
awaiting_voice_command = True
continue_conversation = True
trading_context = False

# Инициализация модели Vosk
model = vosk.Model(r"d:\\model\\vosk-model-small-ru-0.22")


def get_openai_response(prompt):
    logging.info(f"Отправка запроса в OpenAI: {prompt}")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=450
        )
        answer = response.choices[0].message["content"].strip()
        logging.info(f"Ответ от OpenAI: {answer}")
        return answer
    except Exception as e:
        logging.error(f"Ошибка при работе с OpenAI: {e}")
        return "Извините, произошла ошибка."


# Функция для воспроизведения звука
def play_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        if interrupted:
            pygame.mixer.music.stop()
            break
        pygame.time.Clock().tick(10)

# Функция для прерывания
def toggle_interrupt():
    global interrupted, awaiting_voice_command
    interrupted = True
    awaiting_voice_command = True
    logging.info("Прерывание активировано.")

# Слушаем нажатие клавиши space
def check_for_space():
    while True:
        if keyboard.is_pressed('space'):
            toggle_interrupt()
            while keyboard.is_pressed('space'):
                pass

# Функция для распознавания речи
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        logging.info("Слушаю...")
        play_sound('start_listening.mp3')
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        logging.info(f"Вы сказали: {text}")
        return text
    except sr.UnknownValueError:
        logging.warning("Не удалось распознать речь")
        return ""
    except sr.RequestError as e:
        logging.error(f"Ошибка запроса к Google Speech Recognition: {e}")
        return ""

# Асинхронная функция преобразования текста в речь
async def text_to_speech(text):
    global interrupted
    interrupted = False
    play_sound('start_speaking.mp3')

    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "tts-1",
        "voice": "onyx"
    }
    logging.info(f"Отправка запроса на преобразование текста в речь: {text}")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(response.content)
            temp_audio_file_path = temp_audio_file.name
        logging.info(f"text to speexh {temp_audio_file_path}")

        pygame.mixer.music.load(temp_audio_file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            if interrupted:
                pygame.mixer.music.stop()
                break
            pygame.time.Clock().tick(10)
    except requests.RequestException as e:
        logging.error(f"Ошибка преобразования текста в речь: {e}")
#
import MetaTrader5 as mt5
import numpy as np
import logging

# Настройка MetaTrader 5 для получения данных
def get_market_data(symbol, timeframe, num_bars):
    """
    Получение рыночных данных для анализа.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        logging.error(f"Ошибка при получении данных для {symbol}")
        return None
    return rates

# Определение ключевых зон (максимумы и минимумы)
def find_key_zones(rates):
    """
    Нахождение ключевых зон ликвидности на основе экстремумов свечей.
    """
    highs = [rate['high'] for rate in rates]
    lows = [rate['low'] for rate in rates]

    # Нахождение максимумов и минимумов
    max_high = max(highs)
    min_low = min(lows)

    return max_high, min_low

def calculate_atr(symbol, timeframe, period=14):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
    if rates is None or len(rates) == 0:
        logging.error(f"Ошибка при получении данных для {symbol}")
        return None

    high_prices = np.array([rate['high'] for rate in rates])
    low_prices = np.array([rate['low'] for rate in rates])
    close_prices = np.array([rate['close'] for rate in rates])

    # Рассчитываем ATR вручную
    tr = np.maximum(high_prices - low_prices, np.maximum(abs(high_prices - close_prices[:-1]), abs(low_prices - close_prices[:-1])))
    atr = np.mean(tr[-period:])  # Среднее за период

    return atr

def get_atr_info(symbol):
    daily_atr = calculate_atr(symbol, mt5.TIMEFRAME_D1)  # Дневной ATR
    m15_atr = calculate_atr(symbol, mt5.TIMEFRAME_M15)    # ATR для 15 минут

    if daily_atr is None or m15_atr is None:
        logging.error(f"Не удалось рассчитать ATR для {symbol}")
        return None

    current_price = mt5.symbol_info_tick(symbol).bid
    if current_price is None:
        logging.error(f"Не удалось получить информацию о символе {symbol}. Проверьте подключение или правильность символа.")
        return None

    # Рассчитываем данные о прошедших и оставшихся тиках
    daily_atr_ticks = daily_atr / mt5.symbol_info(symbol).point
    m15_atr_ticks = m15_atr / mt5.symbol_info(symbol).point

    atr_info = {
        "daily_atr": daily_atr,
        "m15_atr": m15_atr,
        "daily_atr_ticks": daily_atr_ticks,
        "m15_atr_ticks": m15_atr_ticks,
        "current_price": current_price
    }
    return atr_info

def create_atr_report(atr_info):
    report = (f"Дневной ATR: {atr_info['daily_atr']} ({atr_info['daily_atr_ticks']} тиков)\n"
              f"15-минутный ATR: {atr_info['m15_atr']} ({atr_info['m15_atr_ticks']} тиков)\n"
              f"Текущая цена: {atr_info['current_price']}")
    return report

def create_technical_report(symbol):
    atr_info = get_atr_info(symbol)
    if atr_info is None:
        return "Не удалось получить данные ATR."

    report = create_atr_report(atr_info)
    return report

def summarize_technical_data(result):
    recommendations = []

    if result['atr']['daily_atr'] > 20:
        recommendations.append("Сильный дневной тренд, учитывайте ATR.")
    if result['atr']['m15_atr'] > 5:
        recommendations.append("Возможны сильные колебания в краткосрочной перспективе.")

    detailed_analysis = f"Дневной ATR: {result['atr']['daily_atr']}, 15-минутный ATR: {result['atr']['m15_atr']}"

    return recommendations, detailed_analysis

# Определение зон дисбаланса
def find_imbalance_zones(rates):
    """
    Поиск зон дисбаланса (разрывы между свечами).
    """
    imbalances = []
    for i in range(1, len(rates)):
        if rates[i]['low'] > rates[i-1]['high']:  # Bullish imbalance
            imbalances.append((rates[i-1]['high'], rates[i]['low']))
        elif rates[i]['high'] < rates[i-1]['low']:  # Bearish imbalance
            imbalances.append((rates[i]['low'], rates[i-1]['high']))

    return imbalances

# Пример вызова функций
symbol = "XAUUSDm"
timeframe = mt5.TIMEFRAME_M15
num_bars = 100

# Получаем данные с MetaTrader 5
market_data = get_market_data(symbol, timeframe, num_bars)

if market_data:
    max_high, min_low = find_key_zones(market_data)
    imbalances = find_imbalance_zones(market_data)

    print(f"Ключевые зоны для {symbol}: Максимум: {max_high}, Минимум: {min_low}")
    print(f"Зоны дисбаланса: {imbalances}")


# Функция для технического анализа (определение зоны спроса и предложения)
def detect_supply_demand_zone(symbol, timeframe):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    if rates is None or len(rates) == 0:
        logging.error(f"Ошибка при получении данных для {symbol}")
        return None, None  # Возвращаем None для обоих значений

    highs = [rate['high'] for rate in rates]
    lows = [rate['low'] for rate in rates]

    demand_zone = min(lows)  # Зона спроса
    supply_zone = max(highs)  # Зона предложения

    return supply_zone, demand_zone
import numpy as np
def calculate_rsi(symbol, period=14):
    """
    Рассчитывает RSI для указанного символа.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if rates is None or len(rates) == 0:
        logging.error(f"Ошибка при получении данных для {symbol}")
        return None

    close_prices = [rate['close'] for rate in rates]

    # Рассчитываем RSI с помощью ta-lib
    rsi = talib.RSI(np.array(close_prices), timeperiod=period)
    return rsi[-1]  # Возвращаем последнее значение RSI

def get_rsi(symbol, timeframe, period):
    """
    Функция для расчета RSI для заданного символа и таймфрейма.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
    if rates is None or len(rates) == 0:
        logging.error(f"Ошибка при получении данных для {symbol}")
        return None

    # Извлекаем цены закрытия
    close_prices = np.array([rate['close'] for rate in rates])

    # Рассчитываем изменения цен
    delta = np.diff(close_prices)

    # Вычисляем прибыль и убытки
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # Рассчитываем среднюю прибыль и средний убыток
    avg_gain = np.mean(gain)
    avg_loss = np.mean(loss)

    if avg_loss == 0:
        return 100

    # Рассчитываем относительную силу (RS)
    rs = avg_gain / avg_loss

    # Рассчитываем RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

async def smc_trading_strategy(symbol, volume):
    supply_zone, demand_zone = detect_supply_demand_zone(symbol, mt5.TIMEFRAME_M15)

    # Проверяем, удалось ли получить зоны спроса и предложения
    if supply_zone is None or demand_zone is None:
        logging.error(f"Не удалось получить зоны спроса и предложения для {symbol}. Проверьте символ.")
        await text_to_speech(f"Ошибка: не удалось выполнить анализ для {symbol}. Проверьте символ.")
        return

    current_price = mt5.symbol_info_tick(symbol).bid
    positions = mt5.positions_get(symbol=symbol)

    # Вход в длинную позицию (BUY) с комментарием
    if current_price < demand_zone and len(positions) == 0:
        reason = f"Цена ниже зоны спроса ({demand_zone})"
        open_position(symbol, volume, mt5.ORDER_TYPE_BUY, reason)
        await text_to_speech(f"Открыта длинная позиция для {symbol}, причина: {reason}")

    # Вход в короткую позицию (SELL) с комментарием
    elif current_price > supply_zone and len(positions) == 0:
        reason = f"Цена выше зоны предложения ({supply_zone})"
        open_position(symbol, volume, mt5.ORDER_TYPE_SELL, reason)
        await text_to_speech(f"Открыта короткая позиция для {symbol}, причина: {reason}")

    # Закрытие позиции с комментарием
    for position in positions:
        if position.type == mt5.ORDER_TYPE_BUY and current_price > supply_zone:
            reason = f"Цена достигла зоны предложения ({supply_zone})"
            close_position(position, reason)
            await text_to_speech(f"Закрыта длинная позиция для {symbol}, причина: {reason}")
        elif position.type == mt5.ORDER_TYPE_SELL and current_price < demand_zone:
            reason = f"Цена достигла зоны спроса ({demand_zone})"
            close_position(position, reason)
            await text_to_speech(f"Закрыта короткая позиция для {symbol}, причина: {reason}")


# Пример функции для вычисления MACD
def calculate_macd(symbol, fast_period=12, slow_period=26, signal_period=9):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if rates is None or len(rates) == 0:
        logging.error(f"Ошибка при получении данных для {symbol}")
        return None

    # Извлечение закрывающих цен
    close_prices = [rate['close'] for rate in rates]

    # Вычисление EMA
    fast_ema = talib.EMA(np.array(close_prices), timeperiod=fast_period)
    slow_ema = talib.EMA(np.array(close_prices), timeperiod=slow_period)
    macd = fast_ema - slow_ema
    signal = talib.EMA(macd, timeperiod=signal_period)

    return macd, signal

# Пример функции для вычисления Bollinger Bands
def calculate_bollinger_bands(symbol, period=20):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if rates is None or len(rates) == 0:
        logging.error(f"Ошибка при получении данных для {symbol}")
        return None

    close_prices = [rate['close'] for rate in rates]

    # Вычисляем скользящее среднее и стандартное отклонение
    sma = talib.SMA(np.array(close_prices), timeperiod=period)
    upper_band, middle_band, lower_band = talib.BBANDS(np.array(close_prices), timeperiod=period)

    return upper_band, middle_band, lower_band

def calculate_volume(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if rates is None or len(rates) == 0:
        logging.error(f"Ошибка при получении данных для {symbol}")
        return None

    volumes = [rate['real_volume'] for rate in rates]

    return np.mean(volumes)

def calculate_fibonacci(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if rates is None or len(rates) == 0:
        logging.error(f"Ошибка при получении данных для {symbol}")
        return None

    high_price = max(rate['high'] for rate in rates)
    low_price = min(rate['low'] for rate in rates)

    fib_levels = {
        '23.6%': high_price - 0.236 * (high_price - low_price),
        '38.2%': high_price - 0.382 * (high_price - low_price),
        '50%': high_price - 0.5 * (high_price - low_price),
        '61.8%': high_price - 0.618 * (high_price - low_price)
    }

    return fib_levels

def calculate_atr(symbol, period=7):
    """
    Рассчитывает ATR для указанного символа за последние N дней.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 100)
    if rates is None or len(rates) == 0:
        logging.error(f"Ошибка при получении данных для {symbol}")
        return None

    high_prices = np.array([rate['high'] for rate in rates])
    low_prices = np.array([rate['low'] for rate in rates])
    close_prices = np.array([rate['close'] for rate in rates])

    # Рассчитываем ATR с помощью ta-lib
    atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
    return atr[-1]  # Возвращаем последнее значение ATR


def calculate_stochastic(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if rates is None or len(rates) == 0:
        logging.error(f"Ошибка при получении данных для {symbol}")
        return None

    high_prices = [rate['high'] for rate in rates]
    low_prices = [rate['low'] for rate in rates]
    close_prices = [rate['close'] for rate in rates]

    slowk, slowd = talib.STOCH(np.array(high_prices), np.array(low_prices), np.array(close_prices))

    return slowk[-1], slowd[-1]


def perform_multi_indicator_analysis(symbol):
    logging.info(f"Выполняем многосторонний анализ для {symbol}")

    # Вычисляем MACD
    macd, signal = calculate_macd(symbol)
    logging.info(f"MACD: {macd}, Signal: {signal}")

    # Вычисляем Bollinger Bands
    upper_band, middle_band, lower_band = calculate_bollinger_bands(symbol)
    logging.info(f"Bollinger Bands: Верхняя: {upper_band}, Средняя: {middle_band}, Нижняя: {lower_band}")

    # Вычисляем RSI
    rsi = calculate_rsi(symbol)
    logging.info(f"RSI: {rsi}")

    # Вычисляем Volume
    avg_volume = calculate_volume(symbol)
    logging.info(f"Средний объем: {avg_volume}")

    # Возвращаем результаты для отправки в OpenAI
    return {
        "MACD": macd,
        "Signal": signal,
        "Upper Band": upper_band,
        "Middle Band": middle_band,
        "Lower Band": lower_band,
        "RSI": rsi,
        "Average Volume": avg_volume
    }
def analyze_combined_indicators(symbol):
    atr_info = get_atr_info(symbol)
    macd, signal = calculate_macd(symbol)
    rsi = calculate_rsi(symbol)
    stochastic = calculate_stochastic(symbol)
    fib_levels = calculate_fibonacci(symbol)

    # Начало вывода анализа
    combined_report = f"Технический анализ для {symbol}:\n\n"

    # ATR
    if atr_info:
        combined_report += f"Дневной ATR: {atr_info['daily_atr']} тиков\n"
        combined_report += f"15-минутный ATR: {atr_info['m15_atr']} тиков\n"
    else:
        combined_report += "ATR данные недоступны.\n"

    # MACD
    if macd is not None:
        if macd[-1] > signal[-1]:
            combined_report += "MACD показывает восходящий тренд.\n"
        else:
            combined_report += "MACD показывает нисходящий тренд.\n"

    # RSI
    if rsi:
        if rsi > 70:
            combined_report += "RSI: Перекупленность. Возможен разворот или коррекция.\n"
        elif rsi < 30:
            combined_report += "RSI: Перепроданность. Возможен рост.\n"
        else:
            combined_report += f"RSI: Нейтральная зона ({rsi}).\n"

    # Стохастик
    if stochastic:
        k, d = stochastic
        combined_report += f"Стохастик: K = {k}, D = {d}\n"

    # Фибоначчи
    if fib_levels:
        combined_report += f"Уровни Фибоначчи: 23.6%: {fib_levels['23.6%']}, 38.2%: {fib_levels['38.2%']}, 50%: {fib_levels['50%']}, 61.8%: {fib_levels['61.8%']}\n"

    return combined_report
def generate_recommendations(symbol):
    combined_report = analyze_combined_indicators(symbol)
    recommendations = []

    # Пример рекомендаций на основе индикаторов
    if "восходящий тренд" in combined_report and "RSI: Нейтральная зона" in combined_report:
        recommendations.append("Рассмотрите возможность покупки, восходящий тренд и RSI в нейтральной зоне.")
    if "нисходящий тренд" in combined_report and "RSI: Перекупленность" in combined_report:
        recommendations.append("Рассмотрите возможность продажи, нисходящий тренд и RSI в зоне перекупленности.")

    return recommendations, combined_report

def analyze_multiple_symbols(symbols):
    full_report = ""
    for symbol in symbols:
        recommendations, analysis = generate_recommendations(symbol)
        full_report += f"--- {symbol} ---\n"
        full_report += analysis + "\n"
        full_report += f"Рекомендации: {', '.join(recommendations)}\n\n"

    return full_report

def analyze_combined_indicators(symbol):
    atr_info = get_atr_info(symbol)
    macd, signal = calculate_macd(symbol)
    rsi = calculate_rsi(symbol)
    stochastic = calculate_stochastic(symbol)
    fib_levels = calculate_fibonacci(symbol)

    # Начало вывода анализа
    combined_report = f"Технический анализ для {symbol}:\n\n"

    # ATR
    if atr_info:
        combined_report += f"Дневной ATR: {atr_info['daily_atr']} тиков\n"
        combined_report += f"15-минутный ATR: {atr_info['m15_atr']} тиков\n"
    else:
        combined_report += "ATR данные недоступны.\n"

    # MACD
    if macd is not None:
        if macd[-1] > signal[-1]:
            combined_report += "MACD показывает восходящий тренд.\n"
        else:
            combined_report += "MACD показывает нисходящий тренд.\n"

    # RSI
    if rsi:
        if rsi > 70:
            combined_report += "RSI: Перекупленность. Возможен разворот или коррекция.\n"
        elif rsi < 30:
            combined_report += "RSI: Перепроданность. Возможен рост.\n"
        else:
            combined_report += f"RSI: Нейтральная зона ({rsi}).\n"

    # Стохастик
    if stochastic:
        k, d = stochastic
        combined_report += f"Стохастик: K = {k}, D = {d}\n"

    # Фибоначчи
    if fib_levels:
        combined_report += f"Уровни Фибоначчи: 23.6%: {fib_levels['23.6%']}, 38.2%: {fib_levels['38.2%']}, 50%: {fib_levels['50%']}, 61.8%: {fib_levels['61.8%']}\n"

    return combined_report

def analyze_with_openai(result):
    """
    Отправляет данные анализа в OpenAI для получения прогноза, используя chat-комплит endpoint.
    """
    messages = [
        {"role": "system", "content": "You are a financial analyst assistant and trader."},
        {"role": "user", "content": (
            f"Проанализируй следующие данные для {result['symbol']}:\n"
            f"Зона предложения: {result['supply_zone']}, Зона спроса: {result['demand_zone']}\n"
            f"RSI: {result['rsi']}\n"
            f"MACD: {result['macd'][-1] if result['macd'] is not None else 'N/A'}, "
            f"Signal: {result['signal'][-1] if result['signal'] is not None else 'N/A'}\n"
            f"Bollinger Bands: Верхняя: {result['upper_band'][-1] if result['upper_band'] is not None else 'N/A'}, "
            f"Нижняя: {result['lower_band'][-1] if result['lower_band'] is not None else 'N/A'}\n"
            f"ATR: {result['atr']}\n"
            f"Стохастик: {result['stochastic']}\n"
            f"Уровни Фибоначчи: {result['fibonacci_levels']}\n"
            f"На основе этих данных дай краткий прогноз и рекомендации."
        )}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Используем chat-модель GPT-4
            messages=messages,
            max_tokens=500
        )
        analysis = response['choices'][0]['message']['content'].strip()
        logging.info(f"Ответ от AI: {analysis}")
        return analysis
    except Exception as e:
        logging.error(f"Ошибка при работе с OpenAI: {e}")
        return "Не удалось получить анализ от OpenAI"

def find_order_blocks(rates):
    """
    Определение блоков ордеров на основе движений цены.
    """
    order_blocks = []
    for i in range(1, len(rates)):
        # Поиск крупных движений и их анализ
        if rates[i]['close'] > rates[i]['open'] and rates[i]['high'] > rates[i-1]['high']:
            order_blocks.append(("Bullish Order Block", rates[i]['open'], rates[i]['high']))
        elif rates[i]['close'] < rates[i]['open'] and rates[i]['low'] < rates[i-1]['low']:
            order_blocks.append(("Bearish Order Block", rates[i]['open'], rates[i]['low']))

    return order_blocks

# Функция для проведения технического анализа
import talib

async def run_technical_analysis(symbols):
    results = []
    for symbol in symbols:
        result = technical_analysis(symbol)
        results.append(result)

        # Формируем и отправляем краткое сообщение с результатами анализа
        send_telegram_message(result)

        # Проговариваем результаты анализа голосом
        short_message = create_short_telegram_message(result)
        await text_to_speech(short_message)

    return results





def technical_analysis(symbol):
    logging.info(f"Анализируем {symbol} по стратегии ICT...")

    # 1. Анализ по SMC (Supply/Demand Zones)
    supply_zone, demand_zone = detect_supply_demand_zone(symbol, mt5.TIMEFRAME_M15)
    if supply_zone and demand_zone:
        logging.info(f"SMC для {symbol}: Зона предложения: {supply_zone}, Зона спроса: {demand_zone}")
    else:
        logging.warning(f"Не удалось получить зоны спроса и предложения для {symbol}")

    # 2. Определение ключевых зон ликвидности (Liquidity Zones)
    market_data = get_market_data(symbol, mt5.TIMEFRAME_M15, 100)

    # Исправленная проверка данных
    if market_data is not None and len(market_data) > 0:
        max_high, min_low = find_key_zones(market_data)
        logging.info(f"Ключевые зоны ликвидности для {symbol}: Максимум: {max_high}, Минимум: {min_low}")
    else:
        logging.warning(f"Не удалось получить рыночные данные для {symbol}")

    # 3. Поиск зон дисбаланса (Imbalance Zones)
    imbalances = find_imbalance_zones(market_data)
    if imbalances:
        logging.info(f"Зоны дисбаланса для {symbol}: {imbalances}")
    else:
        logging.warning(f"Не найдены зоны дисбаланса для {symbol}")

    # 4. Order Blocks (блоки ордеров)
    order_blocks = find_order_blocks(market_data)
    if order_blocks:
        logging.info(f"Order Blocks для {symbol}: {order_blocks}")
    else:
        logging.warning(f"Order Blocks не найдены для {symbol}")

    # 5. Уровни Фибоначчи
    fibonacci_levels = calculate_fibonacci(symbol)
    if fibonacci_levels:
        logging.info(f"Уровни Фибоначчи для {symbol}: {fibonacci_levels}")
    else:
        logging.warning(f"Не удалось рассчитать уровни Фибоначчи для {symbol}")

    # 6. MACD
    macd, signal = calculate_macd(symbol)
    logging.info(f"MACD для {symbol}: {macd}, Signal: {signal}")

    # 7. RSI
    rsi_value = calculate_rsi(symbol)
    logging.info(f"RSI для {symbol}: {rsi_value}")

    # 8. Bollinger Bands
    upper_band, middle_band, lower_band = calculate_bollinger_bands(symbol)
    logging.info(f"Bollinger Bands для {symbol}: Верхняя: {upper_band}, Средняя: {middle_band}, Нижняя: {lower_band}")

    # 9. ATR (7 дней)
    atr_value = calculate_atr(symbol, 7)
    logging.info(f"ATR для {symbol}: {atr_value}")

    # 10. Stochastic Oscillator
    stochastic_value = calculate_stochastic(symbol)
    logging.info(f"Стохастик для {symbol}: {stochastic_value}")

    # Возвращаем результат анализа
    return {
        "symbol": symbol,
        "supply_zone": supply_zone,
        "demand_zone": demand_zone,
        "max_high": max_high,
        "min_low": min_low,
        "imbalances": imbalances,
        "order_blocks": order_blocks,
        "fibonacci_levels": fibonacci_levels,  # Добавляем уровни Фибоначчи в результат
        "macd": macd,
        "signal": signal,
        "rsi": rsi_value,
        "upper_band": upper_band,
        "middle_band": middle_band,
        "lower_band": lower_band,
        "atr": atr_value,
        "stochastic": stochastic_value
    }


def openai_analysis_explanation(result):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Убедитесь, что используете правильную модель
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant."},
                {"role": "user", "content": f"Проанализируй эти данные: {result}"}
            ]
        )
        explanation = response['choices'][0]['message']['content']
        return explanation
    except Exception as e:
        logging.error(f"Ошибка при работе с OpenAI: {e}")
        return "Не удалось получить анализ от OpenAI"


# Определение команды для торговли
def detect_trading_command(command):
    command = command.lower()
    if any(word in command for word in trading_keywords['buy']):
        asset = next((asset for asset in trading_keywords['gold'] if asset in command), "XAUUSDm")
        return "buy", asset
    elif any(word in command for word in trading_keywords['sell']):
        asset = next((asset for asset in trading_keywords['gold'] if asset in command), "XAUUSDm")
        return "sell", asset
    elif any(word in command for word in trading_keywords['account']):
        return "account_status", None
    return "unknown_command", None

# Обработка команд торговли
async def handle_trading_command(command):
    action, asset = detect_trading_command(command)
    if action == "buy" and asset:
        await smc_trading_strategy(asset, 0.01)
        await text_to_speech(f"Позиция на покупку для {asset} открыта.")
    elif action == "sell" and asset:
        await text_to_speech(f"Позиция на продажу для {asset} закрыта.")
    elif action == "account_status":
        account_info = get_account_info()
        if account_info:
            await text_to_speech(f"Информация о счете: {account_info}")
    elif "технический анализ" in command.lower():
        symbols = ["XAUUSDm"]
        await text_to_speech("Запускаю технический анализ.")
        await run_technical_analysis(symbols)
def connect_to_mt5():
    # Попытка подключения к MetaTrader 5
    if initialize_mt5():
        account_info = get_account_info()
        if account_info:
            logging.info(f"Подключено к торговому счету. Информация о счете: {account_info}")
            return account_info
        else:
            logging.error("Не удалось получить информацию о счете.")
            return None
    else:
        logging.error("Не удалось подключиться к торговому счету.")
        return None
def handle_command(command):
    global is_active, interrupted, continue_conversation, trading_context
    logging.info(f"Обработка команды: {command}")

    if "создай изображение" in command.lower():
        # Обработка команды создания изображения
        image_description = command.lower().split("создай изображение")[1].strip()
        if image_description:
            img = get_image_from_dalle(image_description)
            if img:
                logging.info("Изображение создано и показано.")
            else:
                logging.warning("Не удалось создать изображение.")
        else:
            logging.warning("Необходимо указать описание изображения после команды 'создай изображение'.")

    elif "отбой" in command.lower():
        is_active = False
        continue_conversation = False  # Завершаем диалог
        logging.info("Ассистент приостановлен.")
        asyncio.run(text_to_speech("До свидания."))

    elif "трейдинг" in command.lower():
        trading_context = True  # Включаем контекст торговли
        account_info = connect_to_mt5()
        if account_info:
            asyncio.run(text_to_speech(f"Подключен к торговому счету. {account_info}"))
        else:
            asyncio.run(text_to_speech("Не удалось подключиться к торговому счету."))
        asyncio.run(text_to_speech("Что бы вы хотели сделать с торговыми позициями?"))

    elif trading_context:
        asyncio.run(handle_trading_command(command))  # Внутри контекста трейдинга

    else:
        response = get_openai_response(command)
        logging.info(f"Джафар: {response}")
        asyncio.run(text_to_speech(response))

market_data_cache = {}

async def get_cached_market_data(symbol):
    if symbol in market_data_cache:
        return market_data_cache[symbol]
    data = await get_market_data(symbol)
    market_data_cache[symbol] = data
    return data

import requests
import logging
def create_short_telegram_message(result):
    """
    Создает краткое сообщение для отправки в Telegram, включая прогноз от OpenAI.
    """
    # Формируем сообщение с ключевыми значениями
    message = (f"Результаты анализа для {result['symbol']}:\n"
               f"Зона предложения: {result['supply_zone']}\n"
               f"Зона спроса: {result['demand_zone']}\n"
               f"RSI: {result['rsi']}\n"
               f"MACD: {result['macd'][-1] if result['macd'] is not None else 'N/A'}, "
               f"Signal: {result['signal'][-1] if result['signal'] is not None else 'N/A'}\n"
               f"Bollinger Bands: Верхняя: {result['upper_band'][-1] if result['upper_band'] is not None else 'N/A'}, "
               f"Нижняя: {result['lower_band'][-1] if result['lower_band'] is not None else 'N/A'}\n"
               f"ATR: {result['atr']}\n"
               f"Стохастик: {result['stochastic']}\n")

    # Проверка наличия уровней Фибоначчи
    if 'fibonacci_levels' in result and result['fibonacci_levels']:
        message += f"Уровни Фибоначчи: {result['fibonacci_levels']}\n"

    # Получаем анализ от OpenAI
    openai_analysis = analyze_with_openai(result)
    message += f"\nПрогноз от OpenAI: {openai_analysis}"

    return message


def create_forecast(result):
    """
    Создает краткий прогноз на основе анализа.
    """
    # Простой пример прогнозирования
    if result['rsi'] > 70:
        return "Перекупленность. Возможен разворот или коррекция."
    elif result['rsi'] < 30:
        return "Перепроданность. Возможен рост."
    else:
        return "Нейтральная зона. Следует продолжать мониторинг."

def send_telegram_message(result):
    """
    Отправляет краткое сообщение в Telegram с результатами анализа.
    """
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        logging.error("Токен бота или ID чата Telegram не установлены.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # Создаем краткое сообщение
    message = create_short_telegram_message(result)

    data = {
        'chat_id': chat_id,
        'text': message
    }

    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        logging.info(f"Сообщение отправлено в Telegram: {message}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка при отправке сообщения в Telegram: {e}")


# Функция для прерывания нажатия клавиши Home
def check_for_home():
    while True:
        if keyboard.is_pressed('home'):  # Меняем проверку на клавишу 'home'
            toggle_interrupt()
            while keyboard.is_pressed('home'):
                pass

# Функция для прерывания
def toggle_interrupt():
    global interrupted, awaiting_voice_command
    interrupted = True
    awaiting_voice_command = True
    logging.info("Прерывание активировано.")

# Функция для обработки прерывания программы
def signal_handler(sig, frame):
    logging.info("Завершение работы...")
    sys.exit(0)

# Функция для обнаружения ключевого слова (например, "Джафар")
def detect_keyword():
    logging.info("Запуск Vosk для обнаружения ключевого слова...")
    rec = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if "джафар" in result['text'].lower():
                    logging.info("Ключевое слово обнаружено. Активирую ассистента...")
                    break
    except KeyboardInterrupt:
        logging.info("Остановка потока обнаружения ключевого слова")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# Главная функция программы
def main():
    global is_active, interrupted, awaiting_voice_command, continue_conversation
    signal.signal(signal.SIGINT, signal_handler)
    logging.info("Готов к работе! Нажмите HOME для отвлечения или скажите 'Джафар' для активации.")

    # Запуск прослушивания нажатия клавиши home
    threading.Thread(target=check_for_home, daemon=True).start()

    while True:
        if continue_conversation:  # Продолжаем диалог после ответа
            logging.info("Ассистент активен. Слушаю команду...")
            command = recognize_speech()
            if command:
                handle_command(command)  # Вызов функции для обработки команд
            interrupted = False  # Сбрасываем флаг прерывания
        elif awaiting_voice_command:  # Ассистент ожидает голосовую команду после активации
            logging.info("Ассистент активен. Слушаю команду...")
            command = recognize_speech()
            if command:
                handle_command(command)  # Вызов функции для обработки команд
            awaiting_voice_command = False  # Завершаем режим ожидания после обработки команды
            interrupted = False  # Сбрасываем флаг прерывания
            logging.info("Ассистент приостановлен. Нажмите HOME или скажите 'Джафар' для повторной активации.")
        else:
            detect_keyword()

if __name__ == "__main__":
    main()
