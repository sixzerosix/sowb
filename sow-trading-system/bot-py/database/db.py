import aiosqlite


async def init_db(db_path="data/trades.db"):
    db = await aiosqlite.connect(db_path)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL,
        quantity REAL NOT NULL,
        pnl REAL,
        status TEXT NOT NULL,
        signal_type TEXT,
        leverage REAL NOT NULL,
        maker_commission REAL NOT NULL,
        taker_commission REAL NOT NULL,
        tp_pct REAL,
        sl_pct REAL,
        tp_price REAL,
        sl_price REAL
    )
    """
    await db.execute(create_table_query)
    await db.commit()
    return db


async def log_trade(db, trade):
    query = """
    INSERT INTO trades (
        timestamp, symbol, timeframe, side, entry_price, exit_price,
        quantity, pnl, status, signal_type, leverage, maker_commission,
        taker_commission, tp_pct, sl_pct, tp_price, sl_price
    ) VALUES (
        :timestamp, :symbol, :timeframe, :side, :entry_price, :exit_price,
        :quantity, :pnl, :status, :signal_type, :leverage, :maker_commission,
        :taker_commission, :tp_pct, :sl_pct, :tp_price, :sl_price
    )
    """
    cursor = await db.execute(query, trade)
    await db.commit()
    return cursor.lastrowid


async def close_trade(db, trade_id, exit_price, pnl):
    query = """
    UPDATE trades 
    SET exit_price = :exit_price, pnl = :pnl, status = 'CLOSED' 
    WHERE id = :id
    """
    await db.execute(query, {"exit_price": exit_price, "pnl": pnl, "id": trade_id})
    await db.commit()
