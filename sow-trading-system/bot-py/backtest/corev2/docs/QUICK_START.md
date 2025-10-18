# ⚡ Быстрая настройка за 5 минут

## 1️⃣ Файлы для изменения

Все настройки находятся в одном файле: **`backtest_optana_v4.py`**

```
backtest/
└── core/
    └── backtest_optana_v4.py  ← ВСЯ НАСТРОЙКА ЗДЕСЬ
```

---

## 2️⃣ Основные настройки (начало файла)

### Режим работы

```python
# Строка ~45
RUN_MODE = "SINGLE"    # или "OPTIMIZE"
```

- **"SINGLE"** - запустить одну стратегию
- **"OPTIMIZE"** - найти лучшие параметры

### Выбор стратегии

```python
# Строка ~48
TARGET_CONFIG_NAME = "EMA_RSI_ATR_Strategy"
```

### Метрика для оптимизации

```python
# Строка ~51
OPTIMIZATION_METRIC = "Sharpe Ratio"
```

Варианты: `"Total PnL"`, `"Return (%)"`, `"Sharpe Ratio"`, `"Max Drawdown (%)"`

---

## 3️⃣ Параметры для оптимизации

```python
# Строка ~60
PARAMETER_SPACE = {
    "EMA_TREND": {
        "fast_len": [5, 9, 13],        # Перебираем эти значения
        "slow_len": [21, 34, 50],
    },
    "RSI": {
        "rsi_len": [14, 20],
    },
    "EXIT": {
        "target_roi_percent": [0.5, 1.0],
        "risk_roi_percent": [0.5, 0.8],
    },
}
```

**Важно**: Система перебирает ВСЕ комбинации (3×3×2×2×2 = 72 варианта)

---

## 4️⃣ Создание своей стратегии

```python
# Строка ~75
STRATEGY_CONFIGS = {
    "EMA_RSI_ATR_Strategy": StrategyConfig(...),  # Существующая
    
    # ДОБАВЬТЕ СВОЮ:
    "My_Strategy": StrategyConfig(
        initial_capital=10000.0,        # Начальный капитал
        leverage=10.0,                   # Плечо
        target_roi_percent=0.8,          # Take Profit %
        risk_roi_percent=0.5,            # Stop Loss %
        indicator_set={
            "EMA_TREND": {
                "fast_len": 9, 
                "slow_len": 21
            },
            "RSI": {
                "rsi_len": 14,
                "overbought": 70,
                "oversold": 30
            },
        },
    ),
}
```

Затем установите:
```python
TARGET_CONFIG_NAME = "My_Strategy"
```

---

## 5️⃣ Параметры стратегии - что они означают

### Основные параметры

| Параметр | Что делает | Рекомендуемые значения |
|----------|------------|------------------------|
| `initial_capital` | Начальный капитал | 1000-10000 USD |
| `leverage` | Плечо (множитель) | 5-20x |
| `target_roi_percent` | Take Profit от цены входа | 0.5-2.0% |
| `risk_roi_percent` | Stop Loss от цены входа | 0.3-1.0% |

### Индикаторы

#### EMA_TREND (Скользящие средние)
```python
"EMA_TREND": {
    "fast_len": 9,      # Быстрая линия (5-20)
    "slow_len": 21,     # Медленная линия (20-100)
}
```
- **Вход Long**: Когда быстрая EMA пересекает медленную ВВЕРХ
- **Вход Short**: Когда быстрая EMA пересекает медленную ВНИЗ

#### RSI (Индекс силы)
```python
"RSI": {
    "rsi_len": 14,       # Период расчета (7-30)
    "overbought": 70,    # Перекупленность (65-80)
    "oversold": 30,      # Перепроданность (20-35)
}
```
- **Long**: Когда RSI < 30 (рынок перепродан)
- **Short**: Когда RSI > 70 (рынок перекуплен)

#### ATR_EXIT (Средний истинный диапазон)
```python
"ATR_EXIT": {
    "atr_len": 14,          # Период ATR (10-20)
    "atr_multiplier": 1.5,  # Множитель для стопов (1.0-3.0)
}
```
- Используется для динамических Stop Loss

---

## 6️⃣ Примеры готовых конфигураций

### 🏃 Агрессивный скальпинг

```python
"Aggressive_Scalper": StrategyConfig(
    initial_capital=10000.0,
    leverage=20.0,              # Высокое плечо
    target_roi_percent=0.3,     # Узкий TP
    risk_roi_percent=0.2,       # Узкий SL
    indicator_set={
        "EMA_TREND": {"fast_len": 5, "slow_len": 13},  # Быстрые EMA
        "RSI": {"rsi_len": 5},                          # Быстрый RSI
    },
)
```

**Когда использовать**: Высокая волатильность, короткий таймфрейм (1m-5m)

### 🐢 Консервативная торговля

```python
"Conservative": StrategyConfig(
    initial_capital=10000.0,
    leverage=5.0,               # Низкое плечо
    target_roi_percent=2.0,     # Широкий TP
    risk_roi_percent=1.0,       # Широкий SL
    indicator_set={
        "EMA_TREND": {"fast_len": 21, "slow_len": 50},  # Медленные EMA
        "RSI": {"rsi_len": 14},
        "MACD": {"fast_len": 12, "slow_len": 26, "signal_len": 9},
    },
)
```

**Когда использовать**: Низкая волатильность, длинный таймфрейм (15m-1h)

### 💥 Пробой волатильности

```python
"Breakout": StrategyConfig(
    initial_capital=10000.0,
    leverage=10.0,
    target_roi_percent=1.5,
    risk_roi_percent=0.8,
    indicator_set={
        "EMA_TREND": {"fast_len": 13, "slow_len": 34},
        "ATR_EXIT": {"atr_len": 14, "atr_multiplier": 2.0},
        "BOLLINGER_BANDS": {"bb_len": 20, "num_dev": 3.0},
    },
)
```

**Когда использовать**: Ожидается сильное движение после консолидации

---

## 7️⃣ Запуск

### Одиночный прогон

```python
RUN_MODE = "SINGLE"
TARGET_CONFIG_NAME = "My_Strategy"
```

```bash
python -m backtest.core.backtest_optana_v4
```

### Оптимизация

```python
RUN_MODE = "OPTIMIZE"
TARGET_CONFIG_NAME = "My_Strategy"
OPTIMIZATION_METRIC = "Sharpe Ratio"

# Настройте пространство поиска
PARAMETER_SPACE = {
    "EMA_TREND": {
        "fast_len": [5, 9, 13],
        "slow_len": [21, 34, 50],
    },
    "RSI": {
        "rsi_len": [14, 20],
    },
}
```

```bash
python -m backtest.core.backtest_optana_v4
```

---

## 8️⃣ Интерпретация результатов

### Что смотреть в первую очередь

```
┌─────────────────────────────────────────┐
│       Метрики Бэктеста                  │
├─────────────────────┬───────────────────┤
│ Total Trades        │ 156               │ ← Должно быть > 30
│ Final Equity        │ $12,450.50        │
│ Total PnL           │ $2,450.50         │ ← Должен быть > 0
│ Return (%)          │ 24.51%            │ ← Хорошо если > 10%
│ Success Rate (%)    │ 58.33%            │ ← Хорошо если > 50%
│ Profit Factor       │ 1.85              │ ← Хорошо если > 1.5
│ Sharpe Ratio        │ 1.42              │ ← Хорошо если > 1.0
│ Max Drawdown (%)    │ 12.34%            │ ← Должно быть < 20%
└─────────────────────┴───────────────────┘
```

### 🚨 Красные флаги

- ❌ **Total Trades < 30**: Недостаточно данных для выводов
- ❌ **Success Rate < 40%**: Стратегия проигрывает чаще
- ❌ **Max Drawdown > 30%**: Слишком рискованная стратегия
- ❌ **Sharpe Ratio < 0.5**: Риск не оправдан доходностью

### ✅ Хорошие показатели

- ✅ **Total Trades > 50**
- ✅ **Success Rate > 55%**
- ✅ **Profit Factor > 1.8**
- ✅ **Sharpe Ratio > 1.5**
- ✅ **Max Drawdown < 15%**

---

## 9️⃣ Типичные ошибки и решения

### ❌ "Нет сделок"

**Проблема**: `Total Trades = 0`

**Решения**:
1. Ослабьте условия входа:
   ```python
   "RSI": {"overbought": 60, "oversold": 40}  # Было 70/30
   ```
2. Уменьшите Stop Loss:
   ```python
   risk_roi_percent=0.3  # Было 0.5
   ```

### ❌ "Слишком много убыточных сделок"

**Проблема**: `Success Rate < 40%`

**Решения**:
1. Увеличьте Take Profit:
   ```python
   target_roi_percent=1.5  # Было 0.8
   ```
2. Используйте более длинные периоды EMA:
   ```python
   "EMA_TREND": {"fast_len": 13, "slow_len": 34}  # Было 9/21
   ```

### ❌ "Большая просадка"

**Проблема**: `Max Drawdown > 25%`

**Решения**:
1. Уменьшите плечо:
   ```python
   leverage=5.0  # Было 10.0
   ```
2. Ужесточьте Stop Loss:
   ```python
   risk_roi_percent=0.3  # Было 0.5
   ```

---

## 🔟 Чеклист настройки

### Перед первым запуском

- [ ] Файл данных существует и содержит > 1000 строк
- [ ] Установлены библиотеки: `pip install pandas numpy numba talib rich matplotlib`
- [ ] Выбран `RUN_MODE` ("SINGLE" или "OPTIMIZE")
- [ ] Указан `TARGET_CONFIG_NAME`

### Для одиночного прогона

- [ ] Настроены параметры стратегии в `STRATEGY_CONFIGS`
- [ ] `initial_capital`, `leverage`, `target_roi_percent`, `risk_roi_percent` установлены
- [ ] Выбраны индикаторы в `indicator_set`

### Для оптимизации

- [ ] Настроено `PARAMETER_SPACE` с разумными диапазонами
- [ ] Выбрана `OPTIMIZATION_METRIC`
- [ ] Количество комбинаций приемлемо (рекомендуется < 500)

---

## 🎓 Полезные советы

### 1. Начните с малого

```python
# Первый запуск - простая стратегия
indicator_set={
    "EMA_TREND": {"fast_len": 9, "slow_len": 21},
    "RSI": {"rsi_len": 14},
}
```

### 2. Постепенно добавляйте сложность

```python
# После успешного теста добавьте фильтры
indicator_set={
    "EMA_TREND": {"fast_len": 9, "slow_len": 21},
    "RSI": {"rsi_len": 14},
    "ATR_EXIT": {"atr_len": 14, "atr_multiplier": 1.5},  # Новое
    "MACD": {"fast_len": 12, "slow_len": 26, "signal_len": 9},  # Новое
}
```

### 3. Сравнивайте разные метрики

Запустите оптимизацию 3 раза с разными метриками:
1. `OPTIMIZATION_METRIC = "Sharpe Ratio"` - лучший баланс
2. `OPTIMIZATION_METRIC = "Total PnL"` - максимальная прибыль
3. `OPTIMIZATION_METRIC = "Max Drawdown (%)"` - минимальный риск

Выберите конфигурацию, которая хороша по всем метрикам.

### 4. Тестируйте на разных периодах

```python
# Тест 1: Бычий рынок (2023 год)
FILE_PATH = "data/BTC_USDT_1m_2023-01-01_to_2023-12-31.csv"

# Тест 2: Медвежий рынок (2022 год)
FILE_PATH = "data/BTC_USDT_1m_2022-01-01_to_2022-12-31.csv"

# Тест 3: Боковик (выберите подходящий период)
FILE_PATH = "data/BTC_USDT_1m_2024-06-01_to_2024-09-01.csv"
```

Хорошая стратегия работает на **всех типах рынка**.

---

## 📞 Нужна помощь?

### Проверьте логи

Система выводит подробную информацию:

```
--- ЗАПУСК СИСТЕМЫ БЭКТЕСТА ---
[LOADER] Успешно загружено 89281 строк данных
--- INFO: Начинаем расчет индикаторов. Исходное количество строк: 89281
--- DEBUG: Рассчитаны EMAs (fast=9, slow=21)
--- DEBUG: Рассчитан RSI (len=14)
--- INFO: Расчет индикаторов завершен. Строк после очистки NaN: 89261
```

Если видите ошибку - прочитайте сообщение, оно укажет на проблему.

---

## 🚀 Быстрый старт (30 секунд)

1. Откройте `backtest_optana_v4.py`
2. Установите:
   ```python
   RUN_MODE = "SINGLE"
   TARGET_CONFIG_NAME = "EMA_RSI_ATR_Strategy"
   ```
3. Запустите:
   ```bash
   python -m backtest.core.backtest_optana_v4
   ```
4. Смотрите результаты! 📊

**Готово!** Теперь вы можете создавать и тестировать свои стратегии! 🎉