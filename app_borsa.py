import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import feedparser
import scipy.optimize as sco
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. CONFIGURAZIONE ---
st.set_page_config(
    page_title="STX Ultimate Suite", 
    page_icon="🏦", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- 2. CSS FIX GRAFICO ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        
        html, body, [class*="css"], .stTextInput, div, span, p {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Nascondiamo completamente il bottone originale buggato */
        [data-testid="stSidebarCollapsedControl"] {
            visibility: hidden !important;
            width: 50px !important; 
            height: 50px !important;
        }

        /* Disegniamo la freccia pulita al suo posto */
        [data-testid="stSidebarCollapsedControl"]::after {
            content: "➤"; 
            visibility: visible !important;
            font-size: 24px !important;
            color: #808080;
            position: absolute;
            top: 20px; left: 20px; /* Posizione fissa */
            cursor: pointer;
        }
        
        [data-testid="stSidebarCollapsedControl"]:hover::after {
            color: #ffffff;
        }

        .big-title { text-align: center; font-size: 3rem !important; font-weight: 700; margin-bottom: 10px; color: var(--text-color); }
        .subtitle { text-align: center; font-size: 1rem; color: gray; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. MENU ---
st.sidebar.title("🕹️ Control Panel")
app_mode = st.sidebar.radio("Modalità:", ["🔎 Analisi Singola (Deep Dive)", "⚖️ Ottimizzatore Portafoglio"])
st.sidebar.markdown("---")
st.sidebar.info("STX Ultimate v4.5\nMath Crash Fix")

# ==============================================================================
# MODALITÀ 1: ANALISI SINGOLA
# ==============================================================================
if app_mode == "🔎 Analisi Singola (Deep Dive)":
    
    st.markdown('<p class="big-title">STX DEEP DIVE</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI + Macro + Monte Carlo + Google News Sentiment</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1: ticker_input = st.text_input("Inserisci Ticker", placeholder="Es. STLA.MI, TSLA").upper().strip()
    with col2: benchmark_input = st.text_input("Benchmark", value="^GSPC").upper().strip()

    # --- NEWS SENTIMENT ---
    def analyze_news_sentiment(ticker):
        try:
            clean_ticker = ticker.split('.')[0]
            rss_url = f"https://news.google.com/rss/search?q={clean_ticker}+stock+market&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            if not feed.entries: return 0, []
            
            analyzer = SentimentIntensityAnalyzer()
            total_score = 0; analyzed_news = []
            panic_words = ["war", "bankrupt", "crash", "crisis", "collapse"]
            hype_words = ["soar", "record", "skyrocket", "surge", "buy"]

            for entry in feed.entries[:10]:
                title = entry.title
                vs = analyzer.polarity_scores(title)
                score = vs['compound']
                if any(w in title.lower() for w in panic_words) and score > -0.5: score = -0.6
                if any(w in title.lower() for w in hype_words) and score < 0.5: score = 0.6
                total_score += score
                color = "#00ff00" if score >= 0.05 else "#ff4444" if score <= -0.05 else "gray"
                analyzed_news.append({'title': title, 'link': entry.link, 'score': score, 'color': color, 'publisher': entry.source.title if 'source' in entry else "Google"})
                
            return (total_score / len(analyzed_news) if analyzed_news else 0), analyzed_news
        except: return 0, []

    # --- DATA FETCHING ---
    @st.cache_data(ttl=12*3600)
    def get_single_data(ticker, benchmark_ticker):
        try:
            stock = yf.Ticker(ticker).history(period="max")
            if stock.empty: stock = yf.download(ticker, period="max", progress=False)
            if stock is None or len(stock) < 300: return None, None, None, None
            
            stock.index = pd.to_datetime(stock.index).tz_localize(None)
            df = stock[['Close']].copy() if 'Close' in stock.columns else stock[['Adj Close']].copy()
            df.rename(columns={df.columns[0]: 'Stock_Price'}, inplace=True)
            
            macro_dict = {"^VIX": "Fear_Index", "GC=F": "Gold_War", "CL=F": "Oil_Energy", "^TNX": "Rates_Inflation"}
            for s, n in macro_dict.items():
                try:
                    m = yf.Ticker(s).history(period="max")
                    m.index = pd.to_datetime(m.index).tz_localize(None)
                    df[n] = m['Close']
                except: df[n] = 0.0
            
            bt = benchmark_ticker if benchmark_ticker else "^GSPC"
            try:
                b = yf.Ticker(bt).history(period="max")
                b.index = pd.to_datetime(b.index).tz_localize(None)
                df['General_Market'] = b['Close']
            except: df['General_Market'] = df['Stock_Price']
            
            df = df.ffill().bfill().dropna(subset=['Stock_Price'])
            if len(df) < 300: return None, None, None, None
            
            df_log = np.log(df / df.shift(1)).fillna(0)
            df_log['Stock_Vol'] = df['Stock_Price'].pct_change().rolling(20).std().fillna(0)
            corr = df_log.iloc[-500:].corr()['Stock_Price'].drop(['Stock_Price', 'Stock_Vol'])
            rs = float(df_log['Stock_Price'].cumsum().iloc[-1] - df_log['General_Market'].cumsum().iloc[-1])
            return df, df_log, corr, rs
        except: return None, None, None, None

    # --- LSTM MODEL ---
    @st.cache_resource(show_spinner=False)
    def train_lstm(df_log):
        cols = ['Stock_Price', 'Fear_Index', 'Gold_War', 'Oil_Energy', 'Rates_Inflation', 'General_Market']
        exist = [c for c in cols if c in df_log.columns]
        data = np.clip(df_log[exist].values, -0.1, 0.1)
        scaler = MinMaxScaler((-1, 1))
        scaled = scaler.fit_transform(data)
        
        X, y = [], []
        PD = 90
        for i in range(PD, len(scaled)):
            X.append(scaled[i-PD:i])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        
        model = Sequential()
        model.add(Input(shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(50, return_sequences=True)); model.add(Dropout(0.2))
        model.add(LSTM(50)); model.add(Dropout(0.2))
        model.add(Dense(25)); model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=15, batch_size=128, verbose=0)
        return model, scaler, scaled, exist

    # --- EXECUTION ---
    if ticker_input:
        progress = st.progress(0, "Analisi in corso...")
        s_score, news = analyze_news_sentiment(ticker_input)
        df_p, df_l, corr, rs = get_single_data(ticker_input, benchmark_input)
        
        if df_p is None:
            st.error("Dati insufficienti."); progress.empty()
        else:
            # Sentiment Display
            st.markdown("##### 📰 Google News Sentiment")
            if s_score > 0.2: l, c, imp = "BULLISH", "#00ff00", 1.05
            elif s_score > 0.05: l, c, imp = "POSITIVO", "#90ee90", 1.02
            elif s_score < -0.2: l, c, imp = "BEARISH", "#ff0000", 0.95
            elif s_score < -0.05: l, c, imp = "NEGATIVO", "#ff4444", 0.98
            else: l, c, imp = "NEUTRALE", "gray", 1.00
            
            c1, c2 = st.columns([1, 2])
            with c1: st.markdown(f"<div style='text-align:center; border:2px solid {c}; padding:10px; border-radius:10px;'><div style='font-size:3rem;'>{s_score:.2f}</div><div style='color:{c}; font-weight:bold;'>{l}</div></div>", unsafe_allow_html=True)
            with c2:
                if news: st.markdown(f"**Top:** [{news[0]['title']}]({news[0]['link']})"); st.caption(f"Fonte: {news[0]['publisher']}")
                else: st.info("Nessuna news recente.")

            # AI Training
            progress.progress(40, "AI Training...")
            model, scaler, scaled, cols = train_lstm(df_l)
            
            # Simulation Loop (IL PUNTO DEL CRASH ERA QUI)
            progress.progress(70, "Simulazione...")
            last_seq = scaled[-90:]
            curr = last_seq.reshape((1, 90, len(cols)))
            fut_ret = []
            macro_trend = np.mean(scaled[-30:, 1:], axis=0) if len(cols)>1 else []
            FD = 365
            
            for i in range(FD):
                p = model.predict(curr, verbose=0)[0, 0]
                p = np.clip(p, -0.05, 0.05) * (0.99**i)
                fut_ret.append(p)
                
                # --- FIX MATEMATICO QUI SOTTO ---
                if len(macro_trend) > 0:
                    mac = macro_trend * (0.95**i) + np.random.normal(0, 0.01, len(macro_trend))
                    new_row = np.insert(mac, 0, p) # Crea array 1D
                else:
                    new_row = np.array([p])
                
                # Reshape forzato per evitare ValueError (Shape Mismatch)
                new_row = new_row.reshape(1, 1, len(new_row)) 
                curr = np.append(curr[:, 1:, :], new_row, axis=1)
                # -------------------------------
                
            dummy = np.zeros((len(fut_ret), len(cols)))
            dummy[:, 0] = fut_ret
            real_ret = scaler.inverse_transform(dummy)[:, 0]
            
            curr_p = df_p['Stock_Price'].iloc[-1]
            fut_p = []
            drift = (imp - 1.0)/30
            for i, r in enumerate(real_ret):
                d = drift if i < 30 else 0
                curr_p *= np.exp(r + d)
                fut_p.append(curr_p)
                
            dates = [df_p.index[-1] + datetime.timedelta(days=i) for i in range(1, FD+1)]
            progress.progress(100, "Fatto."); progress.empty()
            
            # Grafico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p.index[-365:], y=df_p['Stock_Price'].iloc[-365:], name='Storico', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=dates, y=fut_p, name='Forecast AI', line=dict(color='#0055ff', width=3)))
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabs
            t1, t2, t3 = st.tabs(["🧠 Macro", "🔮 Monte Carlo", "🔗 Correlazioni"])
            with t1: st.plotly_chart(go.Figure(go.Bar(x=corr.index, y=corr.values, marker_color=['red' if x<0 else 'green' for x in corr.values])), use_container_width=True)
            with t2:
                u, v = df_l['Stock_Price'].mean(), df_l['Stock_Price'].var()
                dr, sd = u-(0.5*v), df_l['Stock_Price'].std()
                days = np.exp(dr + sd * np.random.normal(0, 1, (FD, 1000)))
                paths = np.zeros_like(days); paths[0] = df_p['Stock_Price'].iloc[-1]
                for t in range(1, FD): paths[t] = paths[t-1] * days[t]
                
                fmc = go.Figure()
                fmc.add_trace(go.Scatter(x=dates, y=np.percentile(paths, 95, axis=1), name='Best', line=dict(color='green')))
                fmc.add_trace(go.Scatter(x=dates, y=np.percentile(paths, 5, axis=1), name='Worst', line=dict(color='red')))
                st.plotly_chart(fmc, use_container_width=True)
            with t3:
                try:
                    bt = benchmark_input if benchmark_input else "^GSPC"
                    bd = yf.Ticker(bt).history(period="max")
                    bd.index = pd.to_datetime(bd.index).tz_localize(None)
                    comb = pd.DataFrame({'A': df_p['Stock_Price'].pct_change(), 'B': bd['Close'].pct_change()}).dropna()
                    roll = comb['A'].rolling(60).corr(comb['B']).dropna()
                    st.line_chart(roll)
                except: st.info("No correlation.")

# ==============================================================================
# MODALITÀ 2: PORTFOLIO OPTIMIZER
# ==============================================================================
elif app_mode == "⚖️ Ottimizzatore Portafoglio":
    st.markdown('<p class="big-title">PORTFOLIO OPTIMIZER</p>', unsafe_allow_html=True)
    def_tickers = "AAPL, MSFT, GOOG, TSLA, STLA.MI, ENI.MI, BTC-USD, GLD"
    tickers_str = st.text_area("Inserisci Ticker", def_tickers, height=70)
    
    c1, c2 = st.columns([1, 2])
    run_opt = c1.button("🚀 Ottimizza", type="primary")
    rf_rate = c2.number_input("Risk-Free Rate", 0.04, step=0.01)

    if run_opt:
        t_list = [x.strip().upper() for x in tickers_str.split(',') if x.strip()]
        if len(t_list) < 2: st.error("Servono almeno 2 asset.")
        else:
            with st.spinner("Calcolo..."):
                closes = pd.DataFrame()
                for t in t_list:
                    try:
                        d = yf.Ticker(t).history(period="2y")
                        if not d.empty:
                            d.index = pd.to_datetime(d.index).tz_localize(None)
                            closes[t] = d['Close']
                    except: pass
                
                closes = closes.ffill().bfill().dropna()
                if closes.shape[1] < 2: st.error("Dati insufficienti."); st.stop()
                
                rets = closes.pct_change(); mean_r = rets.mean(); cov_m = rets.cov()
                num = len(mean_r)
                
                def neg_sharpe(w):
                    r = np.sum(mean_r * w) * 252
                    s = np.sqrt(np.dot(w.T, np.dot(cov_m, w))) * np.sqrt(252)
                    return -(r - rf_rate) / s
                
                cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bnds = tuple((0.0, 1.0) for _ in range(num))
                res = sco.minimize(neg_sharpe, num*[1./num,], method='SLSQP', bounds=bnds, constraints=cons)
                
                st.success("Fatto!")
                labels = closes.columns
                st.plotly_chart(go.Figure(data=[go.Pie(labels=labels, values=res.x, hole=.4)]), use_container_width=True)
