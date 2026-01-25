import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, render_template, jsonify, request
import warnings
from datetime import timedelta, datetime
import math
from flask_cors import CORS

warnings.filterwarnings("ignore")
app = Flask(__name__)

CORS(app)
# --- DATA LOADING ---
try:
    ticker_df = pd.read_csv('yfinance_supported_tickers.csv')
    # adapt to different column names - using what user said earlier
    # expected columns: SYMBOL, NAME OF COMPANY, Yahoo_Equivalent_Code
    rename_map = {}
    if 'NAME OF COMPANY' in ticker_df.columns:
        rename_map['NAME OF COMPANY'] = 'name'
    if 'SYMBOL' in ticker_df.columns:
        rename_map['SYMBOL'] = 'symbol'
    # some users have 'Yahoo_Equivalent_Code' or 'YahooEquiv' etc.
    if 'Yahoo_Equivalent_Code' in ticker_df.columns:
        rename_map['Yahoo_Equivalent_Code'] = 'yfinance_symbol'
    if 'YahooEquiv' in ticker_df.columns:
        rename_map['YahooEquiv'] = 'yfinance_symbol'
    if 'Yahoo_Equivalent' in ticker_df.columns:
        rename_map['Yahoo_Equivalent'] = 'yfinance_symbol'

    ticker_df = ticker_df.rename(columns=rename_map)
    # ensure required columns exist
    for c in ['symbol', 'name', 'yfinance_symbol']:
        if c not in ticker_df.columns:
            ticker_df[c] = None
    ticker_df.dropna(subset=['yfinance_symbol'], inplace=True)
except FileNotFoundError:
    print("FATAL ERROR: 'yfinance_supported_tickers.csv' not found.")
    ticker_df = pd.DataFrame(columns=['symbol', 'name', 'yfinance_symbol'])

# --- HELPER FUNCTIONS ---
def compute_rsi(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, a=12, b=26, c=9):
    """Calculate MACD indicators"""
    ema_short = series.ewm(span=a, adjust=False).mean()
    ema_long = series.ewm(span=b, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=c, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def compute_bollinger_bands(series, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def compute_williams_r(high, low, close, period=14):
    """Calculate Williams %R"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return wr

def compute_commodity_channel_index(high, low, close, period=20):
    """Calculate Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

def get_automated_analysis(info, hist_df):
    """Enhanced automated analysis with comprehensive scoring system."""
    strengths, weaknesses = [], []
    bullish_score = 0
    bearish_score = 0

    # Enhanced Valuation Analysis
    try:
        pe = info.get('trailingPE')
        if pe is not None and isinstance(pe, (int, float)) and pe > 0:
            sector_avg_pe = 25  # Industry average approximation
            if pe < 12:
                strengths.append(f"🎯 Deeply Undervalued: P/E of {pe:.2f} suggests significant undervaluation")
                bullish_score += 3
            elif pe < 18:
                strengths.append(f"💰 Attractive Valuation: P/E of {pe:.2f} indicates potential undervaluation")
                bullish_score += 2
            elif pe < sector_avg_pe:
                strengths.append(f"✅ Reasonable Valuation: P/E of {pe:.2f} is below sector average")
                bullish_score += 1
            elif pe > 45:
                weaknesses.append(f"⚠️ Very High Valuation: P/E of {pe:.2f} indicates extreme premium pricing")
                bearish_score += 3
            elif pe > 30:
                weaknesses.append(f"📈 High Valuation: P/E of {pe:.2f} suggests premium pricing")
                bearish_score += 2
    except Exception:
        pass

    pb = info.get('priceToBook')
    if pb is not None and pb > 0:
        if pb < 1.0:
            strengths.append(f"💎 Trading Below Book Value: P/B ratio of {pb:.2f} suggests deep value opportunity")
            bullish_score += 2
        elif pb < 1.5:
            strengths.append(f"💰 Strong Book Value: P/B ratio of {pb:.2f} indicates good value")
            bullish_score += 1
        elif pb > 8:
            weaknesses.append(f"🔴 Extremely High P/B: {pb:.2f} may indicate severe overvaluation")
            bearish_score += 2
        elif pb > 5:
            weaknesses.append(f"📈 High P/B Ratio: {pb:.2f} may indicate overvaluation")
            bearish_score += 1

    # Enhanced Profitability Analysis
    roe = info.get('returnOnEquity')
    if roe is not None:
        try:
            roe_pct = roe * 100
            if roe_pct > 25:
                strengths.append(f"🚀 Outstanding ROE: {roe_pct:.2f}% indicates exceptional management efficiency")
                bullish_score += 3
            elif roe_pct > 20:
                strengths.append(f"⭐ Excellent ROE: {roe_pct:.2f}% shows superior capital utilization")
                bullish_score += 2
            elif roe_pct > 15:
                strengths.append(f"💪 Strong ROE: {roe_pct:.2f}% indicates good profitability")
                bullish_score += 1
            elif roe_pct < 8:
                weaknesses.append(f"⚡ Weak ROE: {roe_pct:.2f}% suggests poor capital allocation efficiency")
                bearish_score += 2
            elif roe_pct < 12:
                weaknesses.append(f"📉 Below Average ROE: {roe_pct:.2f}% indicates room for improvement")
                bearish_score += 1
        except Exception:
            pass

    pm = info.get('profitMargins')
    if pm is not None:
        try:
            pm_pct = pm * 100
            if pm_pct > 20:
                strengths.append(f"💵 Outstanding Margins: {pm_pct:.2f}% indicates exceptional pricing power")
                bullish_score += 3
            elif pm_pct > 15:
                strengths.append(f"💰 Excellent Margins: {pm_pct:.2f}% shows strong operational efficiency")
                bullish_score += 2
            elif pm_pct > 10:
                strengths.append(f"✅ Healthy Margins: {pm_pct:.2f}% demonstrates good profitability")
                bullish_score += 1
            elif pm_pct < 3:
                weaknesses.append(f"🔴 Very Thin Margins: {pm_pct:.2f}% suggests severe competitive pressure")
                bearish_score += 2
            elif pm_pct < 6:
                weaknesses.append(f"📉 Low Margins: {pm_pct:.2f}% indicates operational challenges")
                bearish_score += 1
        except Exception:
            pass

    # Enhanced Financial Health Analysis
    de = info.get('debtToEquity')
    if de is not None:
        try:
            if de < 25:
                strengths.append(f"🛡️ Very Conservative Debt: D/E ratio of {de:.1f} indicates minimal financial risk")
                bullish_score += 2
            elif de < 50:
                strengths.append(f"✅ Low Debt Levels: D/E ratio of {de:.1f} shows prudent financial management")
                bullish_score += 1
            elif de > 300:
                weaknesses.append(f"🚨 Extremely High Leverage: D/E ratio of {de:.1f} indicates severe financial risk")
                bearish_score += 3
            elif de > 200:
                weaknesses.append(f"⚠️ High Leverage: D/E ratio of {de:.1f} may pose significant financial risk")
                bearish_score += 2
        except Exception:
            pass

    cr = info.get('currentRatio')
    if cr is not None:
        try:
            if cr > 3:
                strengths.append(f"💧 Exceptional Liquidity: Current ratio of {cr:.2f} shows outstanding financial health")
                bullish_score += 2
            elif cr > 2:
                strengths.append(f"💪 Strong Liquidity: Current ratio of {cr:.2f} indicates solid financial position")
                bullish_score += 1
            elif cr < 1:
                weaknesses.append(f"🔴 Poor Liquidity: Current ratio of {cr:.2f} suggests potential cash flow problems")
                bearish_score += 2
            elif cr < 1.2:
                weaknesses.append(f"⚠️ Liquidity Concerns: Current ratio of {cr:.2f} may indicate cash flow challenges")
                bearish_score += 1
        except Exception:
            pass

    # Enhanced Technical Analysis
    try:
        if hist_df is not None and not hist_df.empty and len(hist_df) > 200:
            hist_df = hist_df.copy()
            hist_df['Close'] = hist_df['Close'].astype(float)
            
            # Moving averages
            sma50 = hist_df['Close'].rolling(window=50).mean().iloc[-1]
            sma200 = hist_df['Close'].rolling(window=200).mean().iloc[-1]
            sma20 = hist_df['Close'].rolling(window=20).mean().iloc[-1]
            current_price = hist_df['Close'].iloc[-1]
            
            # Price vs moving averages
            if current_price > sma200 and current_price > sma50 and sma50 > sma200:
                strengths.append(f"📈 Strong Bullish Trend: Golden cross pattern with price above all major SMAs")
                bullish_score += 3
            elif current_price > sma200 and current_price > sma50:
                strengths.append(f"📊 Bullish Momentum: Trading above both 50-day (₹{sma50:.2f}) and 200-day (₹{sma200:.2f}) SMAs")
                bullish_score += 2
            elif current_price > sma200:
                strengths.append(f"📈 Long-term Bullish: Price above 200-day SMA (₹{sma200:.2f})")
                bullish_score += 1
            elif current_price < sma200 and current_price < sma50:
                weaknesses.append(f"📉 Bearish Trend: Trading below major moving averages")
                bearish_score += 2
            elif current_price < sma200:
                weaknesses.append(f"⬇️ Long-term Bearish: Below 200-day SMA (₹{sma200:.2f})")
                bearish_score += 1

            # RSI Analysis
            rsi = compute_rsi(hist_df['Close']).iloc[-1]
            if not np.isnan(rsi):
                if rsi < 25:
                    strengths.append(f"🔥 Severely Oversold: RSI of {rsi:.1f} suggests strong buying opportunity")
                    bullish_score += 2
                elif rsi < 35:
                    strengths.append(f"📉 Oversold Territory: RSI of {rsi:.1f} indicates potential reversal")
                    bullish_score += 1
                elif rsi > 75:
                    weaknesses.append(f"🔴 Severely Overbought: RSI of {rsi:.1f} suggests strong selling pressure")
                    bearish_score += 2
                elif rsi > 65:
                    weaknesses.append(f"⚡ Overbought Conditions: RSI of {rsi:.1f} indicates potential pullback")
                    bearish_score += 1
                
            # MACD Analysis
            macd, signal, histogram = compute_macd(hist_df['Close'])
            macd_current = macd.iloc[-1]
            signal_current = signal.iloc[-1]
            if not (np.isnan(macd_current) or np.isnan(signal_current)):
                if macd_current > signal_current and macd.iloc[-2] <= signal.iloc[-2]:
                    strengths.append(f"📈 MACD Bullish Crossover: Fresh buy signal generated")
                    bullish_score += 2
                elif macd_current > signal_current:
                    strengths.append(f"✅ MACD Bullish: Momentum indicator shows positive trend")
                    bullish_score += 1
                elif macd_current < signal_current and macd.iloc[-2] >= signal.iloc[-2]:
                    weaknesses.append(f"📉 MACD Bearish Crossover: Fresh sell signal generated")
                    bearish_score += 2
                elif macd_current < signal_current:
                    weaknesses.append(f"⬇️ MACD Bearish: Momentum shows negative trend")
                    bearish_score += 1
            
            # Volume Analysis
            avg_volume_20 = hist_df['Volume'].rolling(window=20).mean().iloc[-1]
            recent_volume = hist_df['Volume'].iloc[-5:].mean()
            if recent_volume > avg_volume_20 * 2:
                strengths.append(f"📊 Exceptional Volume Surge: 2x above average indicates strong interest")
                bullish_score += 2
            elif recent_volume > avg_volume_20 * 1.5:
                strengths.append(f"📈 High Volume Activity: Above-average trading suggests increased interest")
                bullish_score += 1
            elif recent_volume < avg_volume_20 * 0.5:
                weaknesses.append(f"📉 Low Volume: Below-average activity suggests lack of interest")
                bearish_score += 1
                
            # Bollinger Bands Analysis
            upper_bb, middle_bb, lower_bb = compute_bollinger_bands(hist_df['Close'])
            bb_upper = upper_bb.iloc[-1]
            bb_lower = lower_bb.iloc[-1]
            if not (np.isnan(bb_upper) or np.isnan(bb_lower)):
                if current_price <= bb_lower:
                    strengths.append(f"📉 Bollinger Band Oversold: Price touching lower band suggests reversal")
                    bullish_score += 1
                elif current_price >= bb_upper:
                    weaknesses.append(f"📈 Bollinger Band Overbought: Price at upper band suggests potential pullback")
                    bearish_score += 1
                
    except Exception as e:
        print(f"Technical analysis error: {e}")

    # Market Position and Growth Analysis
    market_cap = info.get('marketCap')
    if market_cap:
        if market_cap > 200000000000:  # 200B+ Large Cap
            strengths.append("🏢 Mega Cap Stability: Blue-chip stock with market leadership")
            bullish_score += 1
        elif market_cap > 50000000000:  # 50B+ Large Cap
            strengths.append("🏛️ Large Cap Reliability: Established market presence with stability")
        elif market_cap > 5000000000:  # 5B+ Mid Cap
            strengths.append("🎯 Mid Cap Growth Potential: Balance of stability and growth opportunity")

    # Revenue Growth Analysis
    try:
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth is not None:
            growth_pct = revenue_growth * 100
            if growth_pct > 25:
                strengths.append(f"🚀 Exceptional Growth: Revenue growing at {growth_pct:.1f}% annually")
                bullish_score += 3
            elif growth_pct > 15:
                strengths.append(f"📈 Strong Growth: Revenue expanding at {growth_pct:.1f}% per year")
                bullish_score += 2
            elif growth_pct > 8:
                strengths.append(f"📊 Solid Growth: Revenue increasing at {growth_pct:.1f}% annually")
                bullish_score += 1
            elif growth_pct < -10:
                weaknesses.append(f"📉 Revenue Decline: {growth_pct:.1f}% negative growth is concerning")
                bearish_score += 2
            elif growth_pct < 0:
                weaknesses.append(f"⬇️ Revenue Contraction: {growth_pct:.1f}% decline needs attention")
                bearish_score += 1
    except Exception:
        pass

    # Overall Sentiment Scoring with enhanced ranges
    net_score = bullish_score - bearish_score
    if net_score > 6:
        overall_sentiment = "🚀 Very Strongly Bullish"
        sentiment_color = "text-green-400"
    elif net_score > 4:
        overall_sentiment = "📈 Strongly Bullish"
        sentiment_color = "text-green-400"
    elif net_score > 2:
        overall_sentiment = "✅ Moderately Bullish"
        sentiment_color = "text-green-300"
    elif net_score > 0:
        overall_sentiment = "📊 Slightly Bullish"
        sentiment_color = "text-blue-400"
    elif net_score < -6:
        overall_sentiment = "🔴 Very Strongly Bearish"
        sentiment_color = "text-red-400"
    elif net_score < -4:
        overall_sentiment = "📉 Strongly Bearish"
        sentiment_color = "text-red-400"
    elif net_score < -2:
        overall_sentiment = "⬇️ Moderately Bearish"
        sentiment_color = "text-red-300"
    elif net_score < 0:
        overall_sentiment = "📊 Slightly Bearish"
        sentiment_color = "text-orange-400"
    else:
        overall_sentiment = "⚖️ Neutral"
        sentiment_color = "text-yellow-300"

    if not strengths:
        strengths.append("🔍 Limited positive signals detected - consider detailed fundamental research")
    if not weaknesses:
        weaknesses.append("✅ No major red flags identified - continue monitoring key metrics")

    return {
        'strengths': strengths,
        'weaknesses': weaknesses,
        'overall_sentiment': overall_sentiment,
        'sentiment_color': sentiment_color,
        'bullish_score': bullish_score,
        'bearish_score': bearish_score,
        'confidence_level': min(95, max(45, abs(net_score) * 12 + 55))
    }

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '').lower()
    if len(query) < 2 or ticker_df.empty:
        return jsonify([])
    matches = ticker_df[ticker_df['name'].str.lower().str.contains(query, na=False) | ticker_df['symbol'].str.lower().str.contains(query, na=False)]
    # return top 10
    return jsonify(matches.head(10).to_dict(orient='records'))

@app.route('/api/homepage_data')
def homepage_data():
    """Fetch indices and top companies with price, change, percentChange."""
    try:
        tickers_str = "RELIANCE.NS TCS.NS HDFCBANK.NS INFY.NS ICICIBANK.NS HINDUNILVR.NS ITC.NS LT.NS ^NSEI ^NSEBANK ^BSESN"
        tickers = tickers_str.split(' ')
        data = yf.download(tickers, period='5d', progress=False, group_by='ticker', threads=False)
        results = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                # handle multiindex download structure
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.levels[0]:
                        price_info = data[ticker]['Close'].dropna()
                    else:
                        price_info = pd.Series()
                else:
                    # if the API returned flat dataframe for single ticker
                    price_info = data['Close'] if 'Close' in data else pd.Series()
                    if isinstance(price_info, pd.DataFrame) and ticker in price_info.columns:
                        price_info = price_info[ticker].dropna()
                if isinstance(price_info, pd.Series) and len(price_info) > 1:
                    price = float(price_info.iloc[-1])
                    change = float(price - price_info.iloc[-2])
                    percent_change = float((change / price_info.iloc[-2]) * 100)
                else:
                    price = info.get('regularMarketPrice')
                    change = info.get('regularMarketChange')
                    percent_change = info.get('regularMarketChangePercent')
                    if percent_change is not None and abs(percent_change) < 10:  # likely fractional
                        percent_change = percent_change * 100 if abs(percent_change) < 2 else percent_change
                results[ticker] = {"price": price, "change": change, "percentChange": percent_change, "name": info.get('shortName', ticker)}
            except Exception:
                results[ticker] = {"price": None, "change": 0, "percentChange": 0, "name": ticker}
        
        famous_stock_symbols = "RELIANCE.NS TCS.NS HDFCBANK.NS INFY.NS ICICIBANK.NS HINDUNILVR.NS ITC.NS LT.NS".split(' ')
        return jsonify({
            "indices": [{**results.get('^NSEI', {}), "name": "Nifty 50"}, {**results.get('^NSEBANK', {}), "name": "Nifty Bank"}, {**results.get('^BSESN', {}), "name": "Sensex"}],
            "famous_stocks": [{**results.get(sym, {}), "symbol": sym} for sym in famous_stock_symbols]
        })
    except Exception as e:
        return jsonify({"error": f"Failed to fetch homepage data: {e}"}), 500

@app.route('/api/stock/<string:ticker>')
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
            return jsonify({'error': 'Data not available for this ticker.'}), 404

        # historical 3 years
        hist = stock.history(period="3y").reset_index()
        if hist.empty:
            return jsonify({'error': 'Historical data not available.'}), 404
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        
        # No financial statements needed - removed as requested
        financials = {}

        # Enhanced analysis
        analysis = get_automated_analysis(info, hist)

        # reduce info to simple serializable values
        simple_info = {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, type(None)))}

        return jsonify({'info': simple_info, 'historical': hist.to_dict(orient='records'), 'financials': financials, 'analysis': analysis})
    except Exception as e:
        return jsonify({'error': f"An error occurred fetching data for {ticker}: {e}"}), 500

@app.route('/api/intraday/<string:ticker>')
def get_intraday_data(ticker):
    """Endpoint for 1D intraday data at 5m intervals."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="5m").reset_index()
        if hist.empty:
            return jsonify({'error': 'No intraday data'}), 404
        hist.rename(columns={'Datetime': 'Date'}, inplace=True)
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # send only necessary columns
        return jsonify(hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<string:ticker>')
def predict_stock_price(ticker):
    """
    Enhanced rule-based price prediction using comprehensive technical and fundamental analysis
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y").copy()  # Increased to 2 years for better analysis
        info = stock.info
        
        if hist is None or hist.empty or len(hist) < 100:
            return jsonify({'error': 'Insufficient historical data for prediction.'}), 400

        df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        current_price = float(df['Close'].iloc[-1])
        
        # ADVANCED TECHNICAL ANALYSIS SCORING
        technical_score = 0
        technical_factors = []
        
        # 1. Enhanced Moving Average Analysis (Weight: 25%)
        sma_5 = df['Close'].rolling(5).mean().iloc[-1]
        sma_10 = df['Close'].rolling(10).mean().iloc[-1]
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        ema_12 = df['Close'].ewm(span=12).mean().iloc[-1]
        ema_26 = df['Close'].ewm(span=26).mean().iloc[-1]
        
        # Short-term trend (5, 10, 20 day SMAs)
        short_trend_score = 0
        if current_price > sma_5 > sma_10 > sma_20:
            technical_score += 20
            short_trend_score = 20
            technical_factors.append("Strong short-term uptrend (5>10>20 SMA) (+20)")
        elif current_price > sma_10 > sma_20:
            technical_score += 15
            short_trend_score = 15
            technical_factors.append("Moderate short-term uptrend (+15)")
        elif current_price > sma_20:
            technical_score += 10
            short_trend_score = 10
            technical_factors.append("Above 20-day SMA (+10)")
        elif current_price < sma_5 < sma_10 < sma_20:
            technical_score -= 20
            short_trend_score = -20
            technical_factors.append("Strong short-term downtrend (-20)")
        elif current_price < sma_10 < sma_20:
            technical_score -= 15
            short_trend_score = -15
            technical_factors.append("Moderate short-term downtrend (-15)")
        else:
            technical_score -= 5
            short_trend_score = -5
            technical_factors.append("Below 20-day SMA (-5)")
            
        # Medium-term trend (50 day SMA)
        if current_price > sma_50:
            if sma_50 > sma_200:  # Both above long-term
                technical_score += 15
                technical_factors.append("Medium-term bullish: Above 50-day SMA in uptrend (+15)")
            else:
                technical_score += 10
                technical_factors.append("Above 50-day SMA (+10)")
        else:
            if sma_50 < sma_200:  # Both below long-term
                technical_score -= 15
                technical_factors.append("Medium-term bearish: Below 50-day SMA in downtrend (-15)")
            else:
                technical_score -= 10
                technical_factors.append("Below 50-day SMA (-10)")
        
        # Long-term trend (200 day SMA)
        if current_price > sma_200:
            technical_score += 10
            technical_factors.append("Long-term bullish: Above 200-day SMA (+10)")
        else:
            technical_score -= 10
            technical_factors.append("Long-term bearish: Below 200-day SMA (-10)")

        # 2. Enhanced RSI Analysis (Weight: 20%)
        rsi = compute_rsi(df['Close']).iloc[-1]
        rsi_14 = compute_rsi(df['Close'], 14).iloc[-1]
        rsi_21 = compute_rsi(df['Close'], 21).iloc[-1]
        
        if not np.isnan(rsi):
            if rsi < 20:  # Extremely oversold
                technical_score += 25
                technical_factors.append(f"Extremely oversold RSI {rsi:.1f} (+25)")
            elif rsi < 30:  # Oversold
                technical_score += 20
                technical_factors.append(f"Oversold RSI {rsi:.1f} (+20)")
            elif rsi < 40:  # Approaching oversold
                technical_score += 10
                technical_factors.append(f"RSI approaching oversold {rsi:.1f} (+10)")
            elif rsi > 80:  # Extremely overbought
                technical_score -= 25
                technical_factors.append(f"Extremely overbought RSI {rsi:.1f} (-25)")
            elif rsi > 70:  # Overbought
                technical_score -= 20
                technical_factors.append(f"Overbought RSI {rsi:.1f} (-20)")
            elif rsi > 60:  # Approaching overbought
                technical_score -= 10
                technical_factors.append(f"RSI approaching overbought {rsi:.1f} (-10)")
            else:  # Neutral range
                technical_score += 5
                technical_factors.append(f"Neutral RSI {rsi:.1f} (+5)")

        # 3. Advanced MACD Analysis (Weight: 15%)
        macd, signal, histogram = compute_macd(df['Close'])
        macd_current = macd.iloc[-1]
        signal_current = signal.iloc[-1]
        hist_current = histogram.iloc[-1]
        
        if not (np.isnan(macd_current) or np.isnan(signal_current)):
            # MACD Crossovers
            if macd_current > signal_current and macd.iloc[-2] <= signal.iloc[-2]:
                technical_score += 20
                technical_factors.append("Fresh MACD bullish crossover (+20)")
            elif macd_current < signal_current and macd.iloc[-2] >= signal.iloc[-2]:
                technical_score -= 20
                technical_factors.append("Fresh MACD bearish crossover (-20)")
            elif macd_current > signal_current:
                technical_score += 10
                technical_factors.append("MACD above signal line (+10)")
            else:
                technical_score -= 10
                technical_factors.append("MACD below signal line (-10)")
            
            # MACD Histogram momentum
            if hist_current > 0 and histogram.iloc[-2] <= 0:
                technical_score += 10
                technical_factors.append("MACD histogram turning positive (+10)")
            elif hist_current < 0 and histogram.iloc[-2] >= 0:
                technical_score -= 10
                technical_factors.append("MACD histogram turning negative (-10)")

        # 4. Advanced Volume Analysis (Weight: 15%)
        volume_sma_20 = df['Volume'].rolling(20).mean().iloc[-1]
        volume_sma_50 = df['Volume'].rolling(50).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        avg_volume_5 = df['Volume'].iloc[-5:].mean()
        
        # Volume trend analysis
        if avg_volume_5 > volume_sma_20 * 1.5:
            technical_score += 15
            technical_factors.append("High volume confirmation (+15)")
        elif avg_volume_5 > volume_sma_20 * 1.2:
            technical_score += 10
            technical_factors.append("Above average volume (+10)")
        elif avg_volume_5 < volume_sma_20 * 0.7:
            technical_score -= 10
            technical_factors.append("Below average volume (-10)")

        # 5. Bollinger Bands Analysis (Weight: 10%)
        upper_bb, middle_bb, lower_bb = compute_bollinger_bands(df['Close'])
        if not (np.isnan(upper_bb.iloc[-1]) or np.isnan(lower_bb.iloc[-1])):
            bb_position = (current_price - lower_bb.iloc[-1]) / (upper_bb.iloc[-1] - lower_bb.iloc[-1])
            if bb_position < 0.1:  # Near lower band
                technical_score += 15
                technical_factors.append("Near Bollinger lower band - oversold (+15)")
            elif bb_position < 0.2:
                technical_score += 10
                technical_factors.append("Approaching Bollinger lower band (+10)")
            elif bb_position > 0.9:  # Near upper band
                technical_score -= 15
                technical_factors.append("Near Bollinger upper band - overbought (-15)")
            elif bb_position > 0.8:
                technical_score -= 10
                technical_factors.append("Approaching Bollinger upper band (-10)")

        # 6. Momentum Oscillators (Weight: 10%)
        try:
            # Stochastic Oscillator
            k_percent, d_percent = compute_stochastic(df['High'], df['Low'], df['Close'])
            k_current = k_percent.iloc[-1]
            d_current = d_percent.iloc[-1]
            
            if not (np.isnan(k_current) or np.isnan(d_current)):
                if k_current < 20 and d_current < 20:
                    technical_score += 10
                    technical_factors.append("Stochastic oversold condition (+10)")
                elif k_current > 80 and d_current > 80:
                    technical_score -= 10
                    technical_factors.append("Stochastic overbought condition (-10)")
                elif k_current > d_current and k_percent.iloc[-2] <= d_percent.iloc[-2]:
                    technical_score += 8
                    technical_factors.append("Stochastic bullish crossover (+8)")
        except:
            pass

        # 7. Price Action Analysis (Weight: 15%)
        # Recent price momentum
        price_change_1w = (current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100
        price_change_2w = (current_price - df['Close'].iloc[-10]) / df['Close'].iloc[-10] * 100
        price_change_1m = (current_price - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100
        
        if price_change_1w > 5:
            technical_score += 15
            technical_factors.append(f"Strong 1-week momentum +{price_change_1w:.1f}% (+15)")
        elif price_change_1w > 2:
            technical_score += 10
            technical_factors.append(f"Positive 1-week momentum +{price_change_1w:.1f}% (+10)")
        elif price_change_1w < -5:
            technical_score -= 15
            technical_factors.append(f"Weak 1-week momentum {price_change_1w:.1f}% (-15)")
        elif price_change_1w < -2:
            technical_score -= 10
            technical_factors.append(f"Negative 1-week momentum {price_change_1w:.1f}% (-10)")
        
        if price_change_1m > 10:
            technical_score += 10
            technical_factors.append(f"Strong monthly trend +{price_change_1m:.1f}% (+10)")
        elif price_change_1m < -10:
            technical_score -= 10
            technical_factors.append(f"Weak monthly trend {price_change_1m:.1f}% (-10)")

        # ADVANCED FUNDAMENTAL ANALYSIS SCORING
        fundamental_score = 0
        fundamental_factors = []
        
        # 1. Enhanced Valuation Analysis (Weight: 30%)
        pe = info.get('trailingPE')
        if pe and pe > 0:
            # Industry-adjusted P/E analysis
            if pe < 10:
                fundamental_score += 25
                fundamental_factors.append(f"Deeply undervalued P/E {pe:.1f} (+25)")
            elif pe < 15:
                fundamental_score += 20
                fundamental_factors.append(f"Undervalued P/E {pe:.1f} (+20)")
            elif pe < 20:
                fundamental_score += 15
                fundamental_factors.append(f"Fairly valued P/E {pe:.1f} (+15)")
            elif pe < 25:
                fundamental_score += 5
                fundamental_factors.append(f"Reasonable P/E {pe:.1f} (+5)")
            elif pe < 35:
                fundamental_score -= 10
                fundamental_factors.append(f"High P/E {pe:.1f} (-10)")
            else:
                fundamental_score -= 20
                fundamental_factors.append(f"Very high P/E {pe:.1f} (-20)")
        
        # P/B Ratio Analysis
        pb = info.get('priceToBook')
        if pb and pb > 0:
            if pb < 1:
                fundamental_score += 20
                fundamental_factors.append(f"Trading below book value P/B {pb:.1f} (+20)")
            elif pb < 1.5:
                fundamental_score += 15
                fundamental_factors.append(f"Low P/B ratio {pb:.1f} (+15)")
            elif pb < 3:
                fundamental_score += 10
                fundamental_factors.append(f"Reasonable P/B {pb:.1f} (+10)")
            elif pb > 5:
                fundamental_score -= 15
                fundamental_factors.append(f"High P/B ratio {pb:.1f} (-15)")

        # 2. Profitability Analysis (Weight: 25%)
        roe = info.get('returnOnEquity')
        if roe:
            roe_pct = roe * 100
            if roe_pct > 25:
                fundamental_score += 25
                fundamental_factors.append(f"Exceptional ROE {roe_pct:.1f}% (+25)")
            elif roe_pct > 20:
                fundamental_score += 20
                fundamental_factors.append(f"Excellent ROE {roe_pct:.1f}% (+20)")
            elif roe_pct > 15:
                fundamental_score += 15
                fundamental_factors.append(f"Strong ROE {roe_pct:.1f}% (+15)")
            elif roe_pct > 12:
                fundamental_score += 10
                fundamental_factors.append(f"Good ROE {roe_pct:.1f}% (+10)")
            elif roe_pct < 8:
                fundamental_score -= 15
                fundamental_factors.append(f"Weak ROE {roe_pct:.1f}% (-15)")
        
        # Profit Margins
        pm = info.get('profitMargins')
        if pm:
            pm_pct = pm * 100
            if pm_pct > 20:
                fundamental_score += 20
                fundamental_factors.append(f"Excellent margins {pm_pct:.1f}% (+20)")
            elif pm_pct > 15:
                fundamental_score += 15
                fundamental_factors.append(f"Strong margins {pm_pct:.1f}% (+15)")
            elif pm_pct > 10:
                fundamental_score += 10
                fundamental_factors.append(f"Good margins {pm_pct:.1f}% (+10)")
            elif pm_pct < 5:
                fundamental_score -= 10
                fundamental_factors.append(f"Thin margins {pm_pct:.1f}% (-10)")

        # 3. Growth Analysis (Weight: 20%)
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth is not None:
            growth_pct = revenue_growth * 100
            if growth_pct > 25:
                fundamental_score += 25
                fundamental_factors.append(f"Exceptional revenue growth {growth_pct:.1f}% (+25)")
            elif growth_pct > 15:
                fundamental_score += 20
                fundamental_factors.append(f"Strong revenue growth {growth_pct:.1f}% (+20)")
            elif growth_pct > 10:
                fundamental_score += 15
                fundamental_factors.append(f"Good revenue growth {growth_pct:.1f}% (+15)")
            elif growth_pct > 5:
                fundamental_score += 10
                fundamental_factors.append(f"Moderate growth {growth_pct:.1f}% (+10)")
            elif growth_pct < -10:
                fundamental_score -= 20
                fundamental_factors.append(f"Revenue decline {growth_pct:.1f}% (-20)")
            elif growth_pct < 0:
                fundamental_score -= 10
                fundamental_factors.append(f"Revenue contraction {growth_pct:.1f}% (-10)")

        # 4. Financial Health Analysis (Weight: 25%)
        de = info.get('debtToEquity')
        if de is not None:
            if de < 20:
                fundamental_score += 20
                fundamental_factors.append(f"Very low debt D/E {de:.1f} (+20)")
            elif de < 50:
                fundamental_score += 15
                fundamental_factors.append(f"Low debt levels D/E {de:.1f} (+15)")
            elif de < 100:
                fundamental_score += 5
                fundamental_factors.append(f"Moderate debt D/E {de:.1f} (+5)")
            elif de > 200:
                fundamental_score -= 20
                fundamental_factors.append(f"High debt risk D/E {de:.1f} (-20)")
            elif de > 150:
                fundamental_score -= 10
                fundamental_factors.append(f"Elevated debt D/E {de:.1f} (-10)")
        
        cr = info.get('currentRatio')
        if cr is not None:
            if cr > 3:
                fundamental_score += 15
                fundamental_factors.append(f"Excellent liquidity CR {cr:.2f} (+15)")
            elif cr > 2:
                fundamental_score += 10
                fundamental_factors.append(f"Strong liquidity CR {cr:.2f} (+10)")
            elif cr > 1.5:
                fundamental_score += 5
                fundamental_factors.append(f"Adequate liquidity CR {cr:.2f} (+5)")
            elif cr < 1:
                fundamental_score -= 15
                fundamental_factors.append(f"Poor liquidity CR {cr:.2f} (-15)")
            elif cr < 1.2:
                fundamental_score -= 10
                fundamental_factors.append(f"Tight liquidity CR {cr:.2f} (-10)")

        # ADVANCED PREDICTION CALCULATION
        total_score = technical_score + fundamental_score
        
        # Enhanced volatility calculation
        returns = df['Close'].pct_change().dropna()
        daily_volatility = returns.std()
        volatility_factor = min(2.0, max(0.5, daily_volatility * 100))  # Scale volatility
        
        # Advanced trend calculation with multiple timeframes
        short_term_trend = short_trend_score / 20  # Normalize to -1 to 1
        # Advanced trend calculation with multiple timeframes
        short_term_trend = short_trend_score / 20  # Normalize to -1 to 1
        medium_term_trend = (1 if current_price > sma_50 else -1) * 0.7
        long_term_trend = (1 if current_price > sma_200 else -1) * 0.5
        
        # Weighted trend score
        trend_score = (short_term_trend * 0.5 + medium_term_trend * 0.3 + long_term_trend * 0.2)
        
        # Enhanced score-based prediction logic
        if total_score > 80:
            base_multiplier = 1.25  # Very strong bullish
            confidence = "Very High"
        elif total_score > 60:
            base_multiplier = 1.18  # Strong bullish
            confidence = "High"
        elif total_score > 40:
            base_multiplier = 1.12  # Moderately bullish
            confidence = "Medium-High"
        elif total_score > 20:
            base_multiplier = 1.06  # Slightly bullish
            confidence = "Medium"
        elif total_score > -20:
            base_multiplier = 1.00  # Neutral
            confidence = "Low"
        elif total_score > -40:
            base_multiplier = 0.94  # Slightly bearish
            confidence = "Medium"
        elif total_score > -60:
            base_multiplier = 0.88  # Moderately bearish
            confidence = "Medium-High"
        elif total_score > -80:
            base_multiplier = 0.82  # Strong bearish
            confidence = "High"
        else:
            base_multiplier = 0.75  # Very strong bearish
            confidence = "Very High"
        
        # Generate sophisticated predictions
        predictions = []
        upper_band = []
        lower_band = []
        
        # Advanced time-decay and volatility modeling
        for days in range(1, 91):
            # Time decay factor - reduces prediction confidence over time
            time_decay = max(0.3, 1 - (days * 0.012))  # More gradual decay
            
            # Trend persistence factor - trends weaken over time
            trend_persistence = max(0.2, 1 - (days * 0.008))
            
            # Adjusted multiplier with trend and time considerations
            adjusted_multiplier = 1 + (base_multiplier - 1) * time_decay * trend_persistence
            
            # Add cyclical components (quarterly earnings cycles, seasonal effects)
            cyclical_factor = 1 + 0.02 * np.sin(2 * np.pi * days / 90)  # Quarterly cycle
            seasonal_factor = 1 + 0.01 * np.cos(2 * np.pi * days / 365)  # Annual cycle
            
            # Base prediction with multiple factors
            base_prediction = current_price * (adjusted_multiplier ** (days / 30)) * cyclical_factor * seasonal_factor
            
            # Add controlled random walk with mean reversion
            random_component = np.random.normal(0, daily_volatility * 0.3)
            mean_reversion_force = -0.1 * (base_prediction - current_price) / current_price
            
            # Final prediction with bounds checking
            predicted_price = base_prediction * (1 + random_component + mean_reversion_force)
            predicted_price = max(current_price * 0.5, min(current_price * 2.0, predicted_price))  # Reasonable bounds
            
            predictions.append(float(predicted_price))
            
            # Dynamic confidence bands that widen over time
            volatility_expansion = volatility_factor * np.sqrt(days) * 0.01
            confidence_width = predicted_price * volatility_expansion
            
            # Asymmetric bands based on trend direction
            if base_multiplier > 1:  # Bullish trend
                upper_band.append(float(predicted_price + confidence_width * 1.2))
                lower_band.append(float(max(0, predicted_price - confidence_width * 0.8)))
            else:  # Bearish trend
                upper_band.append(float(predicted_price + confidence_width * 0.8))
                lower_band.append(float(max(0, predicted_price - confidence_width * 1.2)))
        
        # Calculate specific time period predictions with enhanced accuracy
        prediction_1w = np.mean(predictions[4:8]) if len(predictions) >= 8 else current_price
        prediction_1m = np.mean(predictions[18:23]) if len(predictions) >= 23 else current_price
        prediction_3m = np.mean(predictions[75:85]) if len(predictions) >= 85 else current_price
        
        # Calculate model confidence based on score consistency
        score_magnitude = abs(total_score)
        score_consistency = min(100, max(50, score_magnitude * 1.2 + 50))
        
        return jsonify({
            "current_price": current_price,
            "prediction_1w": prediction_1w,
            "prediction_1m": prediction_1m,
            "prediction_3m": prediction_3m,
            "full_prediction_series": predictions,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "total_score": total_score,
            "technical_score": technical_score,
            "fundamental_score": fundamental_score,
            "confidence_level": confidence,
            "model_confidence": f"{score_consistency:.0f}",
            "prediction_quality": "High" if score_magnitude > 60 else "Medium" if score_magnitude > 30 else "Low",
            "technical_factors": technical_factors[:8],  # Top 8 factors
            "fundamental_factors": fundamental_factors[:8],  # Top 8 factors
            "prediction_method": "Advanced Rule-Based Analysis",
            "volatility_factor": f"{volatility_factor:.2f}",
            "trend_direction": "Bullish" if base_multiplier > 1.02 else "Bearish" if base_multiplier < 0.98 else "Neutral",
            "top_features": [
                {"feature": "Technical Score", "importance": abs(technical_score) / 100},
                {"feature": "Fundamental Score", "importance": abs(fundamental_score) / 100},
                {"feature": "Price Momentum", "importance": abs(price_change_1m) / 20},
                {"feature": "Volume Trend", "importance": min(1.0, abs(avg_volume_5 / volume_sma_20 - 1))},
                {"feature": "RSI Signal", "importance": abs(50 - rsi) / 50 if not np.isnan(rsi) else 0}
            ]
        })
        
    except Exception as e:
        return jsonify({'error': f"Enhanced prediction analysis failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
