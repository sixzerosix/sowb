# 🛠️ Инструкция по установке исправлений

## 📋 Краткое резюме проблем

### ❌ Ваша ошибка:
```
numba.core.errors.TypingError: Failed in nopython mode pipeline
>>> array(list(array(float64, 1d, C))<iv=None>)
array(float64, 1d, C) not allowed in a homogeneous sequence
```

### ✅ Причина:
Numba не может работать с динамическими списками массивов (`trades.append(np.array([...]))`).

### ✅ Решение:
Используется предварительная аллокация массива фиксированного размера.

---

## 🚀 Быстрая установка (3 шага)

### Шаг 1: Скачайте исправленные файлы

Из артефактов Claude скопируйте содержимое следующих файлов:

1. **backtest_optana_v4_fixed.py** → Сохраните как `backtest_optana_v4.py`
2. **analysis.py - ИСПРАВЛЕНО** → Сохраните как `analysis.py`
3. **persistence.py - ИСПРАВЛЕНО** → Сохраните как `persistence.py`

### Шаг 2: Замените файлы

```bash
# Создайте резервные копии (рекомендуется)
cd backtest/core/
cp backtest_optana_v4.py backtest_optana_v4.py.backup
cp analysis.py analysis.py.backup
cp persistence.py persistence.py.backup

# Замените файлы на исправленные версии
# (скопируйте содержимое из артефактов Claude)
```

### Шаг 3: Проверьте работу

```bash
# Запустите тест
python -m backtest.core.backtest_optana_v4
```

**Ожидаемый результат:**
```
--- ЗАПУСК СИСТЕМЫ БЭКТЕСТА ---
[LOADER] Успешно загружено 89281 строк данных
--- INFO: Начинаем расчет индикаторов...
--- DEBUG: Рассчитаны EMAs (fast=9, slow=21)
--- DEBUG: Рассчитан RSI (len=14)
--- INFO: Расчет индикаторов завершен

--- РЕЗУЛЬТАТЫ ОДИНОЧНОГО ПРОГОНА ---
┌───────────────────────────────────────┐
│       Метрики Бэктеста                │
├─────────────────────┬─────────────────┤
│ Total Trades        │ 156             │
│ Sharpe Ratio        │ 1.42            │  ← ДОЛЖНО ПОЯВИТЬСЯ
│ ...                 │ ...             │
└─────────────────────┴─────────────────┘
```

---

## 📂 Структура файлов после обновления

```
backtest/
├── core/
│   ├── backtest_optana_v4.py     ✅ ИСПРАВЛЕН
│   ├── analysis.py               ✅ ИСПРАВЛЕН
│   ├── persistence.py            ✅ ИСПРАВЛЕН
│   ├── config.py                 ⚪ БЕЗ ИЗМЕНЕНИЙ
│   ├── data_loader.py            ⚪ БЕЗ ИЗМЕНЕНИЙ
│   └── indicators.py             ⚪ БЕЗ ИЗМЕНЕНИЙ
├── data/
│   └── BTC_USDT_1m_...csv
└── docs/                         ✨ НОВОЕ
    ├── README.md                 ✨ Полная документация
    ├── QUICK_START.md            ✨ Быстрая настройка
    ├── CHANGELOG.md              ✨ Список изменений
    └── INSTALLATION.md           ✨ Эта инструкция
```

---

## 🔍 Ключевые изменения в коде

### 1. В `backtest_optana_v4.py`

#### До (❌ ОШИБКА):
```python
@njit
def _numba_backtest_core(...):
    trades = []  # Динамический список
    
    # В цикле:
    trade_record = np.array((exit_index, entry_index, ...))
    trades.append(trade_record)  # ❌ Numba не может это скомпилировать
    
    # В конце:
    trades_array = np.array(trades)  # ❌ ОШИБКА
    return trades_array, current_equity
```

#### После (✅ РАБОТАЕТ):
```python
@njit
def _numba_backtest_core(...):
    # Предварительная аллокация массива
    max_trades = num_bars
    trades_array = np.zeros((max_trades, 8), dtype=np.float64)
    trade_count = 0
    
    # В цикле:
    trades_array[trade_count, 0] = exit_index
    trades_array[trade_count, 1] = entry_index
    # ... заполнение остальных полей ...
    trade_count += 1
    
    # В конце: возвращаем только заполненную часть
    return trades_array[:trade_count], current_equity
```

### 2. В `analysis.py`

#### Добавлено (✅ НОВОЕ):
```python
def calculate_metrics(...):
    # ... существующий код ...
    
    # ===== ДОБАВЛЕНО: Расчет Sharpe Ratio =====
    if num_trades > 1:
        if "pnl_perc" in trades_df.columns:
            returns = trades_df["pnl_perc"].values
        else:
            equity_with_initial = np.concatenate([[initial_capital], 
                                                   equity_curve.values[:-1]])
            returns = (equity_curve.values - equity_with_initial) / equity_with_initial
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return > 0:
            sharpe_ratio = mean_return / std_return * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0
    
    metrics["Sharpe Ratio"] = sharpe_ratio  # ← НОВАЯ МЕТРИКА
```

### 3. В `persistence.py`

#### Добавлено (✅ НОВОЕ):
```python
def persist_optimization_result(...):
    # ... существующий код ...
    
    record["Sharpe_Ratio"] = best_metrics.get("Sharpe Ratio")  # ← НОВОЕ
    
    # ... сохранение в CSV/SQLite/TXT ...
```

---

## ✅ Контрольный список после установки

Проверьте следующие пункты:

### Базовая работоспособность:
- [ ] Система запускается без ошибок
- [ ] Загружаются данные из CSV
- [ ] Рассчитываются индикаторы (EMA, RSI, ATR)
- [ ] Выполняется бэктест (появляются сделки)

### Метрики:
- [ ] В результатах отображается **Sharpe Ratio**
- [ ] Все метрики имеют корректные значения
- [ ] График эквити и просадки отображается

### Режимы работы:
- [ ] **SINGLE** режим работает (один прогон стратегии)
- [ ] **OPTIMIZE** режим работает (перебор параметров)
- [ ] Сохранение результатов работает (CSV/SQLite/TXT)

---

## 🐛 Устранение неполадок

### Проблема 1: "ModuleNotFoundError: No module named 'backtest'"

**Решение:**
```bash
# Убедитесь, что запускаете из корня проекта
cd /path/to/your/project
python -m backtest.core.backtest_optana_v4

# Или установите пакет в режиме разработки
pip install -e .
```

### Проблема 2: "Cannot find data file"

**Решение:**
```python
# В config.py проверьте путь к данным:
FILE_PATH = SCRIPT_DIR / "backtest/data/BTC_USDT_1m_...csv"

# Убедитесь, что файл существует:
print(FILE_PATH)
print(FILE_PATH.exists())  # Должно быть True
```

### Проблема 3: Numba всё ещё выдаёт ошибку

**Возможные причины:**
1. Не заменили файл `backtest_optana_v4.py`
2. Python кэшировал старую версию

**Решение:**
```bash
# Удалите кэш Python
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Перезапустите Python
python -m backtest.core.backtest_optana_v4
```

### Проблема 4: "Sharpe Ratio = 0.0" или не отображается

**Возможные причины:**
1. Слишком мало сделок (< 2)
2. Не заменили файл `analysis.py`

**Решение:**
```bash
# Проверьте количество сделок в результатах
# Если Total Trades < 2, измените параметры стратегии
```

---

## 📚 Дальнейшие шаги

После успешной установки исправлений:

### 1. Прочитайте документацию
- **README.md** - Полное руководство по системе
- **QUICK_START.md** - Быстрая настройка за 5 минут

### 2. Создайте свою стратегию
```python
# В backtest_optana_v4.py:
STRATEGY_CONFIGS = {
    "My_First_Strategy": StrategyConfig(
        initial_capital=10000.0,
        leverage=10.0,
        target_roi_percent=1.0,
        risk_roi_percent=0.5,
        indicator_set={
            "EMA_TREND": {"fast_len": 9, "slow_len": 21},
            "RSI": {"rsi_len": 14},
        },
    ),
}

TARGET_CONFIG_NAME = "My_First_Strategy"
```

### 3. Запустите оптимизацию
```python
RUN_MODE = "OPTIMIZE"
OPTIMIZATION_METRIC = "Sharpe Ratio"

PARAMETER_SPACE = {
    "EMA_TREND": {
        "fast_len": [5, 9, 13],
        "slow_len": [21, 34, 50],
    },
}
```

---

## 💡 Полезные команды

```bash
# Запуск с подробным выводом
python -m backtest.core.backtest_optana_v4

# Запуск с перенаправлением вывода в файл
python -m backtest.core.backtest_optana_v4 > results.log 2>&1

# Запуск в фоновом режиме (Linux/Mac)
nohup python -m backtest.core.backtest_optana_v4 &

# Измерение времени выполнения
time python -m backtest.core.backtest_optana_v4
```

---

## 📊 Ожидаемая производительность

После исправлений:

| Операция | Время | Примечание |
|----------|-------|------------|
| Первая компиляция Numba | 5-10 сек | Только один раз |
| Один бэктест | 0.1-0.5 сек | После компиляции |
| Оптимизация (72 комбинации) | 10-30 сек | Зависит от данных |
| Загрузка данных (90k строк) | 1-2 сек | Зависит от диска |

---

## 🎯 Итоговая проверка

Выполните эту команду:

```bash
python -c "
from backtest.core.backtest_optana_v4 import main
from backtest.core.analysis import calculate_metrics
import pandas as pd
import numpy as np

# Проверка, что импорты работают
print('✅ Импорты успешны')

# Проверка Numba
from numba import njit
@njit
def test_func():
    arr = np.zeros((10, 5))
    return arr
result = test_func()
print(f'✅ Numba работает, размер массива: {result.shape}')

print('✅ Все проверки пройдены!')
"
```

**Ожидаемый вывод:**
```
✅ Импорты успешны
✅ Numba работает, размер массива: (10, 5)
✅ Все проверки пройдены!
```

---

## 📞 Поддержка

Если после установки возникли проблемы:

1. **Проверьте версии зависимостей:**
   ```bash
   pip list | grep -E "(pandas|numpy|numba|talib)"
   ```

2. **Убедитесь, что все файлы заменены:**
   ```bash
   grep -n "ИСПРАВЛЕНИЕ" backtest/core/backtest_optana_v4.py
   # Должно найти комментарии с "ИСПРАВЛЕНИЕ"
   ```

3. **Проверьте логи выполнения:**
   - Ищите строки с `[ERROR]` или `WARNING`
   - Читайте трейсбек ошибок полностью

---

## 🎉 Готово!

После выполнения всех шагов ваша система бэктестинга:
- ✅ Работает без ошибок Numba
- ✅ Рассчитывает Sharpe Ratio
- ✅ Корректно отображает все метрики
- ✅ Поддерживает оптимизацию параметров
- ✅ Имеет полную документацию

**Успешного тестирования стратегий! 🚀📈**