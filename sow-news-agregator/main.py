import asyncio
from playwright.async_api import async_playwright
import pandas as pd
from datetime import datetime
import re
from textblob import TextBlob
from collections import Counter
import json
import aiohttp
from typing import List, Dict, Optional


class AsyncCryptoNewsParser:
    def __init__(self):
        self.sources = {
            "coindesk": {
                "url": "https://www.coindesk.com/",
                "selectors": {
                    "articles": 'div[data-module="ArticleStrip"]',
                    "title": "h3 a",
                    "link": "h3 a",
                    "time": "time",
                },
            },
            "cointelegraph": {
                "url": "https://cointelegraph.com/",
                "selectors": {
                    "articles": ".post-card-inline",
                    "title": ".post-card-inline__title a",
                    "link": ".post-card-inline__title a",
                    "time": ".post-card-inline__date",
                },
            },
            "decrypt": {
                "url": "https://decrypt.co/news",
                "selectors": {
                    "articles": "article",
                    "title": "h2 a, h3 a",
                    "link": "h2 a, h3 a",
                    "time": "time",
                },
            },
            "cryptonews": {
                "url": "https://cryptonews.net/",
                "selectors": {
                    "articles": ".news-item",
                    "title": ".news-title a",
                    "link": ".news-title a",
                    "time": ".news-date",
                },
            },
        }

        self.browser_config = {
            "headless": True,
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

    async def create_browser_context(self, playwright):
        """Создание браузерного контекста"""
        browser = await playwright.chromium.launch(
            headless=self.browser_config["headless"]
        )

        context = await browser.new_context(
            viewport=self.browser_config["viewport"],
            user_agent=self.browser_config["user_agent"],
        )

        return browser, context

    async def parse_source(
        self, source_name: str, source_config: Dict, context
    ) -> List[Dict]:
        """Асинхронный парсинг одного источника"""
        try:
            page = await context.new_page()

            # Блокируем ненужные ресурсы для ускорения
            await page.route(
                "**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}",
                lambda route: route.abort(),
            )

            print(f"🔍 Парсинг {source_name}...")
            await page.goto(
                source_config["url"], wait_until="networkidle", timeout=30000
            )

            # Ждем загрузки контента
            await page.wait_for_timeout(2000)

            articles = []
            selectors = source_config["selectors"]

            # Получаем все статьи
            article_elements = await page.query_selector_all(selectors["articles"])

            for element in article_elements[:20]:  # Берем первые 20 статей
                try:
                    # Извлекаем данные
                    title_elem = await element.query_selector(selectors["title"])
                    title = await title_elem.inner_text() if title_elem else "No title"

                    link_elem = await element.query_selector(selectors["link"])
                    link = await link_elem.get_attribute("href") if link_elem else ""

                    # Обрабатываем относительные ссылки
                    if link and link.startswith("/"):
                        base_url = source_config["url"].rstrip("/")
                        link = base_url + link

                    time_elem = await element.query_selector(selectors["time"])
                    timestamp = (
                        await time_elem.get_attribute("datetime") if time_elem else None
                    )
                    if not timestamp and time_elem:
                        timestamp = await time_elem.inner_text()

                    if title and link:
                        articles.append(
                            {
                                "title": title.strip(),
                                "url": link,
                                "timestamp": timestamp,
                                "source": source_name,
                            }
                        )

                except Exception as e:
                    print(f"⚠️ Ошибка обработки элемента в {source_name}: {e}")
                    continue

            await page.close()
            print(f"✅ {source_name}: собрано {len(articles)} статей")
            return articles

        except Exception as e:
            print(f"❌ Ошибка парсинга {source_name}: {e}")
            return []

    async def get_article_content(self, url: str, context) -> str:
        """Получение полного контента статьи"""
        try:
            page = await context.new_page()
            await page.route(
                "**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}",
                lambda route: route.abort(),
            )

            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_timeout(1000)

            # Универсальные селекторы для контента
            content_selectors = [
                "article",
                ".article-content",
                ".post-content",
                ".entry-content",
                '[data-module="ArticleBody"]',
                ".post__content",
                "main p",
            ]

            content = ""
            for selector in content_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        content = await element.inner_text()
                        if len(content) > 100:  # Минимальная длина контента
                            break
                except:
                    continue

            await page.close()
            return content[:5000]  # Ограничиваем размер

        except Exception as e:
            print(f"⚠️ Ошибка получения контента {url}: {e}")
            return ""

    async def collect_all_news(self) -> List[Dict]:
        """Асинхронный сбор новостей со всех источников"""
        async with async_playwright() as playwright:
            browser, context = await self.create_browser_context(playwright)

            try:
                # Создаем задачи для всех источников
                tasks = []
                for source_name, source_config in self.sources.items():
                    task = self.parse_source(source_name, source_config, context)
                    tasks.append(task)

                # Выполняем все задачи параллельно
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Объединяем результаты
                all_articles = []
                for result in results:
                    if isinstance(result, list):
                        all_articles.extend(result)
                    else:
                        print(f"⚠️ Ошибка в задаче: {result}")

                return all_articles

            finally:
                await context.close()
                await browser.close()

    async def process_articles_batch(
        self, articles: List[Dict], batch_size: int = 10
    ) -> List[Dict]:
        """Обработка статей батчами"""
        async with async_playwright() as playwright:
            browser, context = await self.create_browser_context(playwright)

            try:
                processed_data = []
                analyzer = NewsAnalyzer()

                # Обрабатываем батчами
                for i in range(0, len(articles), batch_size):
                    batch = articles[i : i + batch_size]
                    print(
                        f"📄 Обработка батча {i//batch_size + 1}/{(len(articles)-1)//batch_size + 1}"
                    )

                    # Создаем задачи для получения контента
                    content_tasks = []
                    for article in batch:
                        task = self.get_article_content(article["url"], context)
                        content_tasks.append(task)

                    # Получаем контент параллельно
                    contents = await asyncio.gather(
                        *content_tasks, return_exceptions=True
                    )

                    # Анализируем каждую статью
                    for article, content in zip(batch, contents):
                        if isinstance(content, str):
                            full_text = article["title"] + " " + content

                            analysis = {
                                "title": article["title"],
                                "url": article["url"],
                                "source": article["source"],
                                "timestamp": article.get("timestamp"),
                                "content_length": len(content),
                                "sentiment": analyzer.analyze_sentiment(full_text),
                                "mentioned_cryptos": analyzer.extract_cryptocurrencies(
                                    full_text
                                ),
                                "categories": analyzer.categorize_news(
                                    article["title"], content
                                ),
                                "price_movements": analyzer.extract_price_movements(
                                    content
                                ),
                                "processed_at": datetime.now().isoformat(),
                            }

                            processed_data.append(analysis)

                    # Небольшая пауза между батчами
                    await asyncio.sleep(1)

                return processed_data

            finally:
                await context.close()
                await browser.close()


class NewsAnalyzer:
    def __init__(self):
        self.crypto_keywords = {
            "bitcoin": ["bitcoin", "btc"],
            "ethereum": ["ethereum", "eth", "ether"],
            "ripple": ["ripple", "xrp"],
            "cardano": ["cardano", "ada"],
            "solana": ["solana", "sol"],
            "dogecoin": ["dogecoin", "doge"],
            "binance": ["binance", "bnb"],
            "polygon": ["polygon", "matic"],
            "avalanche": ["avalanche", "avax"],
            "chainlink": ["chainlink", "link"],
        }

        self.market_indicators = {
            "bullish": [
                "bull",
                "bullish",
                "pump",
                "moon",
                "surge",
                "rally",
                "breakout",
                "rise",
            ],
            "bearish": [
                "bear",
                "bearish",
                "dump",
                "crash",
                "fall",
                "decline",
                "drop",
                "correction",
            ],
            "neutral": ["stable", "sideways", "consolidation", "range-bound", "flat"],
        }

    def analyze_sentiment(self, text: str) -> str:
        """Улучшенный анализ тональности"""
        text_lower = text.lower()

        # Подсчет индикаторов
        bullish_score = sum(
            1 for word in self.market_indicators["bullish"] if word in text_lower
        )
        bearish_score = sum(
            1 for word in self.market_indicators["bearish"] if word in text_lower
        )

        # TextBlob анализ
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        # Комбинированная оценка
        if bullish_score > bearish_score and polarity > -0.1:
            return "bullish"
        elif bearish_score > bullish_score and polarity < 0.1:
            return "bearish"
        else:
            return "neutral"

    def extract_cryptocurrencies(self, text: str) -> List[str]:
        """Извлечение криптовалют с учетом контекста"""
        text_lower = text.lower()
        found_cryptos = []

        for crypto_name, keywords in self.crypto_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_cryptos.append(crypto_name.upper())
                    break

        return list(set(found_cryptos))

    def extract_price_data(self, text: str) -> Dict:
        """Извлечение ценовых данных"""
        price_patterns = {
            "prices": r"\$[\d,]+\.?\d*",
            "percentages": r"(\d+\.?\d*)\s*%",
            "market_cap": r"market\s*cap.*?\$[\d,]+\.?\d*[kmb]?",
            "volume": r"volume.*?\$[\d,]+\.?\d*[kmb]?",
        }

        extracted_data = {}
        for key, pattern in price_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted_data[key] = matches[:5]  # Первые 5 совпадений

        return extracted_data


class AsyncCryptoNewsAggregator:
    def __init__(self):
        self.parser = AsyncCryptoNewsParser()
        self.analyzer = NewsAnalyzer()

    async def run_full_analysis(self, max_articles: int = 100):
        """Полный цикл анализа новостей"""
        print("🚀 Запуск асинхронного парсера криптоновостей...")

        # Сбор новостей
        start_time = datetime.now()
        articles = await self.parser.collect_all_news()
        collect_time = datetime.now() - start_time

        print(
            f"📊 Собрано {len(articles)} статей за {collect_time.total_seconds():.1f} сек"
        )

        if not articles:
            print("❌ Статьи не найдены")
            return

        # Ограничиваем количество для обработки
        articles = articles[:max_articles]

        # Обработка и анализ
        start_process = datetime.now()
        processed_data = await self.parser.process_articles_batch(
            articles, batch_size=5
        )
        process_time = datetime.now() - start_process

        print(
            f"⚡ Обработано {len(processed_data)} статей за {process_time.total_seconds():.1f} сек"
        )

        # Генерация отчета
        report = self.generate_comprehensive_report(processed_data)

        # Сохранение
        await self.save_data_async(processed_data, "crypto_news_data.json")
        await self.save_data_async(report, "crypto_news_report.json")

        print("✅ Анализ завершен!")
        self.print_summary(report)

        return processed_data, report

    def generate_comprehensive_report(self, data: List[Dict]) -> Dict:
        """Генерация подробного отчета"""
        df = pd.DataFrame(data)

        return {
            "summary": {
                "total_articles": len(data),
                "processing_time": datetime.now().isoformat(),
                "sources": df["source"].value_counts().to_dict(),
            },
            "sentiment_analysis": {
                "distribution": df["sentiment"].value_counts().to_dict(),
                "by_source": df.groupby("source")["sentiment"].value_counts().to_dict(),
            },
            "cryptocurrency_mentions": {
                "top_mentioned": self.get_top_cryptos(data),
                "by_sentiment": self.get_crypto_sentiment_breakdown(data),
            },
            "categories": df["categories"].explode().value_counts().to_dict(),
            "trends": {
                "most_discussed_topics": self.extract_trending_topics(data),
                "price_movement_frequency": self.analyze_price_movements(data),
            },
        }

    async def save_data_async(self, data, filename: str):
        """Асинхронное сохранение данных"""
        import aiofiles

        async with aiofiles.open(filename, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"💾 Сохранено: {filename}")

    def print_summary(self, report: Dict):
        """Вывод краткого отчета"""
        print("\n📈 КРАТКИЙ ОТЧЕТ:")
        print(f"Статей обработано: {report['summary']['total_articles']}")
        print(f"Источников: {len(report['summary']['sources'])}")
        print(
            f"Топ криптовалют: {list(report['cryptocurrency_mentions']['top_mentioned'].keys())[:5]}"
        )
        print(f"Настроение рынка: {report['sentiment_analysis']['distribution']}")


# Запуск
async def main():
    aggregator = AsyncCryptoNewsAggregator()
    await aggregator.run_full_analysis(max_articles=50)


if __name__ == "__main__":
    asyncio.run(main())
