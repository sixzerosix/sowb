"""
МОДУЛЬ ВИЗУАЛИЗАЦИИ СТАКАНА ЗАКАЗОВ В РЕАЛЬНОМ ВРЕМЕНИ

Основные функции:
1. Получение данных стакана с биржи Binance через WebSocket
2. Группировка ордеров по заданному шагу плотности
3. Интерактивное управление параметрами отображения
4. Цветное форматирование вывода в терминал
"""

import asyncio
import ccxt.async_support as ccxt  # Асинхронная версия CCXT
from decimal import Decimal  # Для точных финансовых вычислений


class AdvancedOrderBookVisualizer:
    def __init__(
        self, symbol, depth=10, width=80, update_interval=1, density_step=None
    ):
        """
        Инициализация параметров визуализатора

        :param symbol: Торговая пара (например, 'BTC/USDT')
        :param depth: Количество отображаемых уровней стакана
        :param width: Ширина вывода в символах
        :param update_interval: Интервал обновления в секундах
        :param density_step: Шаг группировки ордеров (None для сырых данных)
        """
        self.symbol = symbol
        self.depth = depth
        self.width = width
        self.update_interval = update_interval
        self.density_step = Decimal(str(density_step)) if density_step else None
        self.exchange = ccxt.binance(
            {  # Инициализация подключения к бирже
                "enableRateLimit": True,  # Включение встроенного rate limiting
                "options": {"defaultType": "spot"},  # Тип торговли (спот)
            }
        )
        self.running = True  # Флаг для управления циклом выполнения

    async def fetch_data(self):
        """Получение данных стакана с биржи"""
        return await self.exchange.fetch_order_book(
            self.symbol, limit=1000  # Максимальное количество получаемых ордеров
        )

    def calculate_density(self, orders, is_bid=True):
        """
        Группировка ордеров по плотности

        :param orders: Список ордеров вида [[price, amount], ...]
        :param is_bid: Флаг направления (bid/ask)
        :return: Сгруппированные данные
        """
        if not self.density_step:
            # Режим без группировки - возвращаем первые N ордеров
            return [
                (Decimal(price), Decimal(amount))
                for price, amount in orders[: self.depth]
            ]

        grouped = {}
        for price, amount in orders:
            dec_price = Decimal(price)
            step = self.density_step

            # Вычисление уровня группировки
            level = (dec_price // step) * step
            if is_bid:
                level += step  # Для бидов смещаем уровень вверх

            key = float(level)
            grouped[key] = grouped.get(key, Decimal(0)) + Decimal(amount)

        # Сортировка: для бидов - по убыванию, для асков - по возрастанию
        sorted_items = sorted(grouped.items(), key=lambda x: -x[0] if is_bid else x[0])
        return sorted_items[: self.depth]

    def prepare_data(self, orderbook):
        """Подготовка данных для отображения"""
        raw_bids = orderbook["bids"]  # Ордера на покупку
        raw_asks = orderbook["asks"]  # Ордера на продажу

        # Группировка данных
        bids = self.calculate_density(raw_bids, is_bid=True)
        asks = self.calculate_density(raw_asks, is_bid=False)

        return bids, asks

    def format_row(self, bid, ask):
        """
        Форматирование строки вывода для одного уровня стакана

        :param bid: Данные бида (цена, объем)
        :param ask: Данные аска (цена, объем)
        :return: Отформатированная строка с ANSI-цветами
        """
        # Форматирование чисел
        bid_price = f"{bid[0]:.2f}" if bid else ""
        bid_amount = f"{bid[1]:.6f}" if bid else ""
        ask_price = f"{ask[0]:.2f}" if ask else ""
        ask_amount = f"{ask[1]:.6f}" if ask else ""

        # Цветовое оформление: зеленый для бидов, красный для асков
        bid_str = f"\033[92m{bid_price:>10} | {bid_amount:>12}\033[0m"
        ask_str = f"\033[91m{ask_amount:<12} | {ask_price:<10}\033[0m"

        # Информация о диапазонах плотности
        density_info = ""
        if self.density_step:
            step = float(self.density_step)
            # Формирование диапазонов для отображения
            bid_range = f"[{bid[0]-step:.2f}-{bid[0]:.2f}]" if bid else ""
            ask_range = f"[{ask[0]:.2f}-{ask[0]+step:.2f}]" if ask else ""
            density_info = f"{bid_range:^15} || {ask_range:^15}"

        return f"{bid_str} || {ask_str} {density_info}"

    def visualize(self, bids, asks):
        """Отрисовка стакана в терминале"""
        # Очистка экрана (работает в большинстве терминалов)
        print("\033c", end="")

        # Заголовок
        title = f"{self.symbol} Order Book - Density Mode: {self.density_step or 'Raw'}"
        print(title.center(self.width))
        print("=" * self.width)

        # Шапка таблицы
        headers = [
            f"{'BIDS':^{self.width//2}} || {'ASKS':^{self.width//2}}",
            "-" * self.width,
            " Price      | Amount      ||      Amount | Price"
            + ("          Density Ranges" if self.density_step else ""),
        ]
        print("\n".join(headers))
        print("-" * self.width)

        # Основные данные
        max_len = max(len(bids), len(asks))
        for i in range(max_len):
            bid = bids[i] if i < len(bids) else None
            ask = asks[i] if i < len(asks) else None
            print(self.format_row(bid, ask))

        # Подвал
        print("=" * self.width)
        stats = f"Levels: {self.depth} | Update: {self.update_interval}s | Step: {self.density_step or 'N/A'}"
        print(stats.center(self.width))
        print("\nCtrl+C to exit | Change density with: 'd <step>'")

    async def handle_input(self):
        """Обработка пользовательского ввода"""
        while self.running:
            try:
                # Асинхронное чтение ввода
                cmd = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
                if cmd.lower().startswith("d "):
                    # Изменение шага плотности
                    new_step = cmd.split()[1]
                    self.density_step = Decimal(new_step)
                    print(f"\nDensity step changed to {new_step}")
                elif cmd.lower() == "exit":
                    # Корректное завершение
                    self.running = False
            except Exception as e:
                print(f"Input error: {e}")

    async def run(self):
        """Основной цикл выполнения"""
        input_task = asyncio.create_task(self.handle_input())

        try:
            while self.running:
                # Получение и обработка данных
                orderbook = await self.fetch_data()
                bids, asks = self.prepare_data(orderbook)
                self.visualize(bids, asks)

                # Ожидание следующего обновления
                await asyncio.sleep(self.update_interval)
        except KeyboardInterrupt:
            self.running = False
        finally:
            # Корректное завершение
            await self.exchange.close()
            input_task.cancel()


if __name__ == "__main__":
    import sys

    # Фикс для Windows (выбор правильного event loop)
    if sys.platform == "win32":
        from asyncio import WindowsSelectorEventLoopPolicy

        asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

    # Инициализация и запуск
    visualizer = AdvancedOrderBookVisualizer(
        symbol="BTC/USDT",  # Торговая пара
        depth=10,  # Глубина стакана
        width=100,  # Ширина вывода
        update_interval=2,  # Интервал обновления (сек)
        density_step=10,  # Шаг группировки (None для отключения)
    )

    asyncio.run(visualizer.run())
