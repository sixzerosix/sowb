import aiohttp


async def send_telegram_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    params = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params) as response:
                return await response.json()
        except Exception as e:
            print("Ошибка при отправке сообщения в Telegram:", e)
            return None
