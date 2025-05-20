import re
import time
import threading
import asyncio
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from telethon import TelegramClient, events
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import talib
from fastapi.middleware.cors import CORSMiddleware

# === FastAPI Setup ===
app = FastAPI()

# Add CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot_status = {"running": True}

# === Logging Configuration ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Config ===
api_id = 25870667
api_hash = '23096783fce69e0db44b35e6b19adc21'
session_name = 'signal_parser_session'
BOT_TOKEN = '7708673880:AAGhpXVk5np6JoFJmFMW2_75u9KRrYLPBPQ'

# === Trading Parameters ===
INITIAL_LOT = 0.01  # Fixed lot size
RISK_AMOUNT = 2.0  # Euros
PROFIT_TARGET = 0.30  # 30 cents profit target
LAYERING_THRESHOLD = -2.5  # When to add layering position
COMMISSION_PER_LOT = 0.08
COOLDOWN_SECONDS = 1.0

# Trading state variables
last_trade_time = None
trading_enabled = True
open_trades = {}  # Track open trades

# Symbol mapping to standardize broker-specific symbols
SYMBOL_MAPPING = {
    # Gold
    'GOLD': ['XAUUSD', 'XAUUSD+', 'XAUUSD-', 'XAU/USD'],
    # Major FX
    'EURUSD': ['EURUSD', 'EUR/USD', 'EURUSD+', 'EURUSD-'],
    'GBPUSD': ['GBPUSD', 'GBP/USD', 'GBPUSD+', 'GBPUSD-'],
    'USDJPY': ['USDJPY', 'USD/JPY', 'USDJPY+', 'USDJPY-'],
    # Crypto
    'BTCUSD': ['BTCUSD', 'BTC/USD', 'BTCUSD+', 'BTCUSD-'],
    'ETHUSD': ['ETHUSD', 'ETH/USD', 'ETHUSD+', 'ETHUSD-']
}

# Market analysis parameters
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_PERIOD = 14

# === API Endpoints ===
@app.get("/status")
async def get_status():
    return bot_status

@app.post("/pause")
async def pause_bot():
    bot_status["running"] = False
    return {"message": "Bot paused"}

@app.post("/resume")
async def resume_bot():
    bot_status["running"] = True
    return {"message": "Bot resumed"}

class RiskParams(BaseModel):
    risk: float

@app.post("/risk")
async def set_risk(params: RiskParams):
    global RISK_AMOUNT
    RISK_AMOUNT = params.risk
    return {"message": f"Risk set to {params.risk}"}

@app.get("/open_trades")
async def get_open_trades():
    positions = get_positions()
    return {"trades": positions}

@app.post("/close_all")
async def close_all_trades():
    result = close_all_positions()
    return {"message": f"Closed {result} positions"}

# === NEW MANUAL TRADING ENDPOINTS ===
class TradeRequest(BaseModel):
    symbol: str
    action: str  # "buy" or "sell"

@app.post("/manual_trade")
async def manual_trade(request: TradeRequest):
    if not bot_status["running"]:
        return {"success": False, "message": "Bot is paused"}
    
    if request.action not in ["buy", "sell"]:
        return {"success": False, "message": "Invalid action. Use 'buy' or 'sell'"}
    
    # Execute the trade
    ticket = execute_trade(request.action, request.symbol)
    
    if ticket:
        return {
            "success": True, 
            "message": f"{request.action.upper()} order placed for {request.symbol}",
            "ticket": ticket
        }
    else:
        return {"success": False, "message": "Trade execution failed"}

@app.post("/close_trade/{ticket}")
async def close_specific_trade(ticket: int):
    success = close_position(ticket)
    if success:
        return {"success": True, "message": f"Position {ticket} closed"}
    else:
        return {"success": False, "message": f"Failed to close position {ticket}"}

@app.get("/symbols")
async def get_available_symbols():
    symbols = []
    for base_symbol in SYMBOL_MAPPING:
        for variant in SYMBOL_MAPPING[base_symbol]:
            if mt5.symbol_select(variant, True):
                symbols.append(variant)
                break
    
    return {"symbols": symbols}

# === MT5 Connection ===
def connect_mt5():
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return False
    
    logger.info("MT5 connected successfully")
    account_info = mt5.account_info()
    if account_info:
        logger.info(f"Account: {account_info.login}, Balance: {account_info.balance}")
    return True

# === Symbol Detection ===
def detect_symbol(message_text):
    """Detect trading symbol from message text"""
    message_text = message_text.upper()
    
    # Direct symbol mentions
    for base_symbol, variants in SYMBOL_MAPPING.items():
        for variant in variants:
            if variant in message_text:
                return find_valid_symbol(base_symbol)
    
    # Common terms
    if any(term in message_text for term in ['GOLD', 'XAU']):
        return find_valid_symbol('GOLD')
    elif 'BITCOIN' in message_text or 'BTC' in message_text:
        return find_valid_symbol('BTCUSD')
    elif 'EURO' in message_text or 'EUR' in message_text:
        return find_valid_symbol('EURUSD')
    
    # Default to gold if no symbol detected
    return find_valid_symbol('GOLD')

def find_valid_symbol(base_symbol):
    """Find a valid symbol variant in MT5"""
    variants = SYMBOL_MAPPING.get(base_symbol, [base_symbol])
    
    for variant in variants:
        if mt5.symbol_select(variant, True):
            return variant
    
    logger.warning(f"No valid symbol found for {base_symbol}")
    return None

# === Signal Detection ===
def parse_signal(message_text):
    """Parse trading signal from message text"""
    message_text = message_text.lower()
    
    # Match buy signals
    buy_patterns = [
        r'\bbuy\b', r'long\b', r'calls?\b', r'bullish\b', 
        r'buy\s+signal', r'buy\s+now', r'buy\s+gold'
    ]
    
    # Match sell signals
    sell_patterns = [
        r'\bsell\b', r'short\b', r'puts?\b', r'bearish\b',
        r'sell\s+signal', r'sell\s+now', r'sell\s+gold'
    ]
    
    # Check for buy signals
    for pattern in buy_patterns:
        if re.search(pattern, message_text):
            return "buy", extract_symbol(message_text), extract_targets(message_text)
    
    # Check for sell signals
    for pattern in sell_patterns:
        if re.search(pattern, message_text):
            return "sell", extract_symbol(message_text), extract_targets(message_text)
    
    return None, None, None

def extract_symbol(message_text):
    """Extract symbol from message text"""
    return detect_symbol(message_text)

def extract_targets(message_text):
    """Extract TP/SL values from message text"""
    tp_match = re.search(r'tp\s*[:-]?\s*(\d+\.?\d*)', message_text, re.IGNORECASE)
    sl_match = re.search(r'sl\s*[:-]?\s*(\d+\.?\d*)', message_text, re.IGNORECASE)
    
    tp = float(tp_match.group(1)) if tp_match else None
    sl = float(sl_match.group(1)) if sl_match else None
    
    return {"tp": tp, "sl": sl}

# === Trading Functions ===
def in_cooldown():
    """Check if we're in the cooldown period between trades"""
    return last_trade_time and (datetime.now() - last_trade_time).total_seconds() < COOLDOWN_SECONDS

def calculate_lot_size(symbol, risk_amount):
    """Calculate lot size based on risk"""
    # Always use fixed lot size of 0.01 as specified
    return INITIAL_LOT

def execute_trade(action, symbol, sl_override=None, tp_override=None):
    """Execute a trade on MT5"""
    global last_trade_time
    
    if in_cooldown() or not trading_enabled:
        return None
    
    if not symbol or not mt5.symbol_select(symbol, True):
        logger.error(f"Invalid symbol: {symbol}")
        return None
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"Failed to get tick for {symbol}")
        return None
    
    price = tick.ask if action == 'buy' else tick.bid
    
    # Always use fixed lot size of 0.01
    lot_size = INITIAL_LOT
    
    # Create trade request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": 10,
        "magic": 12345,  # Magic number to identify our trades
        "comment": "Signal bot trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    # Add SL/TP if provided
    if sl_override:
        request["sl"] = sl_override
    
    if tp_override:
        request["tp"] = tp_override
    
    # Send the order
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"✅ {action.upper()} {symbol} executed at {price}, lot: {lot_size}")
        last_trade_time = datetime.now()
        
        # Track the trade
        trade_info = {
            "ticket": result.order,
            "symbol": symbol,
            "action": action,
            "lot_size": lot_size,
            "entry_price": price,
            "open_time": datetime.now(),
            "parent_trade": None
        }
        open_trades[result.order] = trade_info
        
        return result.order
    else:
        logger.error(f"❌ Order failed: {result.comment}, code: {result.retcode}")
        return None

def get_positions(symbol=None):
    """Get current open positions"""
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()
    
    if positions is None:
        return []
    
    result = []
    for pos in positions:
        result.append({
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "type": "buy" if pos.type == 0 else "sell",
            "volume": pos.volume,
            "open_price": pos.price_open,
            "current_price": pos.price_current,
            "profit": pos.profit,
            "sl": pos.sl,
            "tp": pos.tp,
            "time": datetime.fromtimestamp(pos.time).strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return result

def close_position(ticket):
    """Close a specific position"""
    positions = mt5.positions_get()
    if not positions:
        return False
    
    for pos in positions:
        if pos.ticket == ticket:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 10,
                "magic": 12345,
                "comment": "Signal bot close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Closed position {ticket}")
                if ticket in open_trades:
                    del open_trades[ticket]
                return True
            else:
                logger.error(f"Failed to close position {ticket}: {result.comment}")
    
    return False

def close_all_positions():
    """Close all open positions"""
    positions = mt5.positions_get()
    if not positions:
        return 0
    
    closed_count = 0
    for pos in positions:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask,
            "deviation": 10,
            "magic": 12345,
            "comment": "Signal bot close all",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            closed_count += 1
            if pos.ticket in open_trades:
                del open_trades[pos.ticket]
    
    open_trades.clear()  # Clear all tracked trades
    return closed_count

# === Market Analysis ===
def get_market_data(symbol, timeframe, bars_count):
    """Get historical market data for analysis"""
    if timeframe == "1m":
        tf = mt5.TIMEFRAME_M1
    elif timeframe == "15m":
        tf = mt5.TIMEFRAME_M15
    else:
        tf = mt5.TIMEFRAME_M1  # Default
    
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars_count)
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to get rates for {symbol} {timeframe}")
        return None
    
    return rates

def calculate_rsi(rates, period=14):
    """Calculate RSI indicator"""
    if rates is None or len(rates) < period+1:
        return None
    
    # Extract close prices
    close_prices = np.array([rate[4] for rate in rates])
    
    # Use talib for accurate RSI calculation
    rsi = talib.RSI(close_prices, timeperiod=period)
    
    # Return the latest RSI value
    return rsi[-1]

def analyze_market(symbol):
    """Analyze market conditions for trading decisions"""
    # Get market data
    m1_data = get_market_data(symbol, "1m", 100)
    m15_data = get_market_data(symbol, "15m", 100)
    
    if m1_data is None or m15_data is None:
        logger.error("Failed to get market data for analysis")
        return None, None
    
    # Calculate indicators
    rsi_m1 = calculate_rsi(m1_data)
    rsi_m15 = calculate_rsi(m15_data)
    
    if rsi_m1 is None or rsi_m15 is None:
        logger.error("Failed to calculate indicators")
        return None, None
    
    # Determine market trend
    m1_trend = "up" if rsi_m1 > 50 else "down"
    m15_trend = "up" if rsi_m15 > 50 else "down"
    
    # Determine signal based on RSI
    signal = None
    if rsi_m1 < RSI_OVERSOLD and rsi_m15 < 45:
        signal = "buy"  # Oversold condition
    elif rsi_m1 > RSI_OVERBOUGHT and rsi_m15 > 55:
        signal = "sell"  # Overbought condition
    
    logger.info(f"Analysis: {symbol} - 1m RSI: {rsi_m1:.2f}, 15m RSI: {rsi_m15:.2f}, Signal: {signal}")
    
    return signal, {"m1_trend": m1_trend, "m15_trend": m15_trend, "rsi_m1": rsi_m1, "rsi_m15": rsi_m15}

# === Position Management ===
def manage_positions():
    """Monitor and manage open positions"""
    while trading_enabled:
        try:
            positions = get_positions()
            
            for position in positions:
                ticket = position["ticket"]
                symbol = position["symbol"]
                profit = position["profit"]
                
                # Auto-close at 30 cent profit target
                if profit >= PROFIT_TARGET:
                    logger.info(f"Position {ticket} hit profit target: {profit}")
                    close_position(ticket)
                
                # Additional layering logic (keep for advanced management)
                if ticket in open_trades:
                    trade_info = open_trades[ticket]
                    
                    # Check if we need layering for drawdown management
                    if profit <= LAYERING_THRESHOLD and not trade_info.get("layered", False):
                        # Add layering trade when we hit threshold
                        action = "buy" if trade_info["action"] == "buy" else "sell"
                        layering_ticket = execute_trade(action, symbol)
                        
                        if layering_ticket:
                            logger.info(f"Added layering position {layering_ticket} for {ticket}")
                            open_trades[layering_ticket]["parent_trade"] = ticket
                            open_trades[ticket]["layered"] = True
                            open_trades[ticket]["layering_trade"] = layering_ticket
            
            # Sleep to avoid excessive CPU usage
            time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error in position management: {e}")
            time.sleep(5)  # Wait longer on error

# === Telethon Client ===
async def setup_telegram_client():
    """Initialize and start the Telegram client"""
    client = TelegramClient(session_name, api_id, api_hash)
    
    @client.on(events.NewMessage)
    async def on_message(event):
        """Handle new messages from any channel"""
        if not bot_status["running"]:
            return
        
        try:
            # Parse the message for trading signals
            message_text = event.message.text
            action, symbol, targets = parse_signal(message_text)
            
            if action and symbol:
                logger.info(f"Signal detected: {action} {symbol}")
                
                # First check if the signal makes sense with our market analysis
                signal, analysis = analyze_market(symbol)
                
                # If analysis confirms signal or is neutral, execute trade
                if signal is None or signal == action:
                    sl_override = None
                    tp_override = None
                    
                    if targets and targets["sl"]:
                        sl_override = targets["sl"]
                    if targets and targets["tp"]:
                        tp_override = targets["tp"]
                    
                    ticket = execute_trade(action, symbol, sl_override, tp_override)
                    if ticket:
                        await event.respond(f"✅ Executed {action.upper()} for {symbol} based on signal")
                else:
                    logger.info(f"Signal {action} contradicts market analysis {signal}, not executing")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    await client.start()
    return client

# === Main Application ===
async def start_bot():
    """Start the trading bot"""
    # Initialize MT5
    if not connect_mt5():
        logger.error("Failed to connect to MT5, exiting")
        return
    
    # Start position management in a background thread
    threading.Thread(target=manage_positions, daemon=True).start()
    
    # Start Telegram client
    client = await setup_telegram_client()
    
    logger.info("Bot started successfully")
    await client.run_until_disconnected()

# Start the bot when the application starts
@app.on_event("startup")
async def on_startup():
    asyncio.create_task(start_bot())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
