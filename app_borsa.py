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
    initial_sidebar_state="collapsed"  # <--- NASCONDE LA TENDINA ALL'AVVIO
)

# --- 2. CSS (FIX FRECCIA & STILE) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        
        /* Font Globale */
        html, body, [class*="css"], .stTextInput, div, span, p {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* FIX SIDEBAR BUTTON (Quello che ti dava problemi) */
        [data-testid="stSidebarCollapsedControl"] {
            color: transparent !important; /* Nasconde la scritta 'keyboard_double...' */
            border: none !important;
        }
        
        [data-testid="stSidebarCollapsedControl"]::after {
            content: "➤"; /* Disegna una freccia pulita */
            color: #808080; /* Colore Grigio */
            font-size: 25px;
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            transition: color 0.3s;
        }

        [data-testid="stSidebarCollapsedControl"]:hover::after {
            color: #ffffff; /* Diventa bianca al passaggio del mouse */
        }

        /* Titoli e Stile Generale */
        .big-title {
            text-align: center; font-size: 3rem !important; font-weight: 700;
            margin-bottom: 10px; color: var(--text-color);
        }
        .subtitle {
            text-align: center; font-size: 1rem; color: gray; margin-bottom: 30px;
        }
        .stButton>button {
            width: 100%; border-radius: 10px; font-weight: bold;
        }
        .news-box {
            border: 1px solid rgba(128,128,128,0.2); border-radius: 10px; 
            padding: 15px; margin-bottom: 10px; background-color: rgba(255,255,255,0.02);
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. MENU LATERALE ---
st.sidebar.title("🕹️ Control Panel")
app_mode = st.sidebar.radio(
    "Modalità:",
    ["🔎 Analisi Singola (Deep Dive)", "⚖️ Ottimizzatore Portafoglio"]
)
st.sidebar.markdown("---")
st.sidebar.info("STX Ultimate v4.1\nAI + Google News + Markowitz")

# ==============================================================================
# MODALITÀ 1: ANALISI SINGOLA (DEEP DIVE)
# ==============================================================================
if app_mode == "🔎 Analisi Singola (Deep Dive)":
    
    st.markdown('<p class="big-title">STX DEEP DIVE</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI + Macro + Monte Carlo + Google News Sentiment</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input("Inserisci Ticker", placeholder="Es. STLA.MI, TSLA, BTC-USD").upper().strip()
    with col2:
        benchmark_input = st.text_input("Benchmark", value="^GSPC", help="Es. ^GSPC, FTSEMIB.MI").upper().strip()

    # --- FUNZIONI ---
    def analyze_news_sentiment(ticker):
        try:
            clean_ticker = ticker.split('.')[0]
            rss_url = f"https://news.google.com/rss/search?q={clean_ticker}+stock+market&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            if not feed.entries: return 0, []
            
            analyzer = SentimentIntensityAnalyzer()
            total_score = 0
            analyzed_news = []
            panic_words = ["war", "bankrupt", "fraud", "crash", "investigation", "crisis", "plunge", "collapse"]
            hype_words = ["soar", "record", "breakthrough", "skyrocket", "jump", "surge", "beats", "buy"]

            for entry in feed.entries[:10]:
                title = entry.title
                link = entry.link
                publisher = entry.source.title if 'source' in entry else "Google News"
                vs = analyzer.polarity_scores(title)
                score = vs['compound']
                
                t_low = title.lower()
                if any(w in t_low for w in panic_words):
                    if score > -0.5: score = -0.6
                elif any(w in t_low for w in hype_words):
                    if score < 0.5: score = 0.6
                    
                total_score += score
                color = "#00ff00" if score >= 0.05 else "#ff4444" if score <= -0.05 else "gray"
                analyzed_news.append({'title': title, 'link': link, 'score': score, 'color': color, 'publisher': publisher})
                
            return (total_score / len(analyzed_news) if analyzed_news else 0), analyzed_news
        except: return 0, []

    @st.cache_data(ttl=12*3600)
    def get_single_data(ticker, benchmark_ticker):
        try:
            stock = yf.Ticker(ticker).history(period="max")
            if stock.empty: stock = yf.download(ticker, period="max", progress=False)
            if stock is None or len(stock) < 300: return None, None, None, None
            
            stock.index = pd.to_datetime(stock.index).tz_localize(None)
            if 'Close' in stock.columns: df = stock[['Close']].copy()
            elif 'Adj Close' in stock.columns: df = stock[['Adj Close']].copy()
            else: return None, None, None, None
            
            df.rename(columns={df.columns[0]: 'Stock_Price'}, inplace=True)
            
            macro_dict = {"^VIX": "Fear_Index", "GC=F": "Gold_War", "CL=F": "Oil_Energy", "^TNX": "Rates_Inflation"}
            for symbol, name in macro_dict.items():
                try:
                    m = yf.Ticker(symbol).history(period="max")
                    m.index = pd.to_datetime(m.index).tz_localize(None)
                    df[name] = m['Close']
                except: df[name] = 0.0
            
            if not benchmark_ticker: benchmark_ticker = "^GSPC"
            try:
                b = yf.Ticker(benchmark_ticker).history(period="max")
                b.index = pd.to_datetime(b.index).tz_localize(None)
                df['General_Market'] = b['Close']
            except:
                try:
                    b = yf.Ticker("^GSPC").history(period="max")
                    b.index = pd.to_datetime(b.index).tz_localize(None)
                    df['General_Market'] = b['Close']
                except: df['General_Market'] = df['Stock_Price']
            
            df = df.ffill().bfill()
            df.dropna(subset=['Stock_Price'], inplace=True)
            if len(df) < 300: return None, None, None, None
            
            df_log = np.log(df / df.shift(1)).fillna(0)
            df_log['Stock_Vol'] = df['Stock_Price'].pct_change().rolling(20).std().fillna(0)
            try:
                rs = float(df_log['Stock_Price'].cumsum().iloc[-1] - df_log['General_Market'].cumsum().iloc[-1])
            except: rs = 0.0
            
            corr = df_log.iloc[-500:].corr()['Stock_Price'].drop(['Stock_Price', 'Stock_Vol'])
            return df, df_log, corr, rs
        except: return None, None, None, None

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

    if ticker_input:
        progress = st.progress(0, "Analisi in corso...")
        s_score, news = analyze_news_sentiment(ticker_input)
        df_p, df_l, corr, rs = get_single_data(ticker_input, benchmark_input)
        
        if df_p is None:
            st.error("Dati insufficienti."); progress.empty()
        else:
            # Display Sentiment
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

            # AI
            progress.progress(40, "AI Training...")
            model, scaler, scaled, cols = train_lstm(df_l)
            
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
                new_row = [p]
                if len(macro_trend)>0:
                    mac = macro_trend*(0.95**i) + np.random.normal(0, 0.01, len(macro_trend))
                    new_row = np.insert(mac, 0, p)
                curr = np.append(curr[:, 1:, :], [new_row], axis=1)
                
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
            
            # Chart Main
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p.index[-365:], y=df_p['Stock_Price'].iloc[-365:], name='Storico', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=dates, y=fut_p, name='Forecast AI', line=dict(color='#0055ff', width=3)))
            st.plotly_chart(fig, use_container_width=True)
            
            chg = ((fut_p[-1] - df_p['Stock_Price'].iloc[-1]) / df_p['Stock_Price'].iloc[-1])*100
            rc = "#00ff00" if rs > 0 else "#ff4444"
            rsign = "+" if rs > 0 else ""
            st.markdown(f"<div style='text-align:center; font-size:1.1rem;'>Target 1Y: <b>{fut_p[-1]:.2f}</b> | Trend: <b style='color:{'#00ff00' if chg>0 else '#ff4444'}'>{chg:+.2f}%</b><br><span style='color:gray; font-size:0.9rem;'>Forza Relativa: <b style='color:{rc}'>{rsign}{rs*100:.2f}%</b></span></div>", unsafe_allow_html=True)
            
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
                fmc.add_trace(go.Scatter(x=dates, y=np.percentile(paths, 50, axis=1), name='Median', line=dict(color='white', dash='dot')))
                st.plotly_chart(fmc, use_container_width=True)
                loss = ((df_p['Stock_Price'].iloc[-1] - np.percentile(paths, 5, axis=1)[-1]) / df_p['Stock_Price'].iloc[-1])*100
                st.error(f"⚠️ VaR (95%): Rischio max stimato -{loss:.2f}%")
            with t3:
                try:
                    bt = benchmark_input if benchmark_input else "^GSPC"
                    bd = yf.Ticker(bt).history(period="max")
                    bd.index = pd.to_datetime(bd.index).tz_localize(None)
                    comb = pd.DataFrame({'A': df_p['Stock_Price'].pct_change(), 'B': bd['Close'].pct_change()}).dropna()
                    roll = comb['A'].rolling(60).corr(comb['B']).dropna()
                    fc = go.Figure()
                    fc.add_trace(go.Scatter(x=roll.index, y=roll.values, line=dict(color='#ff00ff')))
                    fc.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=0.8, x1=1, y1=1, fillcolor="rgba(255,0,0,0.1)", line_width=0)
                    st.plotly_chart(fc, use_container_width=True)
                except: st.info("No correlation.")

# ==============================================================================
# MODALITÀ 2: PORTFOLIO OPTIMIZER
# ==============================================================================
elif app_mode == "⚖️ Ottimizzatore Portafoglio":
    
    st.markdown('<p class="big-title">PORTFOLIO OPTIMIZER</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Efficient Frontier • Markowitz • Asset Allocation</p>', unsafe_allow_html=True)

    def_tickers = "AAPL, MSFT, GOOG, TSLA, STLA.MI, ENI.MI, BTC-USD, GLD"
    tickers_str = st.text_area("Inserisci Ticker (separati da virgola)", def_tickers, height=70)
    
    c_btn, c_risk = st.columns([1, 2])
    with c_btn: run_opt = st.button("🚀 Ottimizza", type="primary")
    with c_risk: rf_rate = st.number_input("Risk-Free Rate", 0.04, step=0.01)

    def port_perf(weights, mean_ret, cov):
        ret = np.sum(mean_ret * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
        return std, ret

    def neg_sharpe(weights, mean_ret, cov, rf):
        std, ret = port_perf(weights, mean_ret, cov)
        return -(ret - rf) / std

    if run_opt:
        t_list = [x.strip().upper() for x in tickers_str.split(',') if x.strip()]
        if len(t_list) < 2: st.error("Inserisci almeno 2 asset.")
        else:
            with st.spinner("Scaricamento e Ottimizzazione..."):
                closes = pd.DataFrame()
                valid = []
                for t in t_list:
                    try:
                        d = yf.Ticker(t).history(period="2y")
                        if not d.empty:
                            d.index = pd.to_datetime(d.index).tz_localize(None)
                            closes[t] = d['Close']
                            valid.append(t)
                    except: st.warning(f"Errore su {t}")
                
                if len(valid) < 2: st.error("Dati insufficienti.")
                else:
                    closes = closes.ffill().bfill().dropna()
                    rets = closes.pct_change()
                    mean_r = rets.mean()
                    cov_m = rets.cov()
                    num = len(valid)
                    
                    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                    bnds = tuple((0.0, 1.0) for _ in range(num))
                    init = num * [1./num,]
                    
                    res = sco.minimize(neg_sharpe, init, args=(mean_r, cov_m, rf_rate), method='SLSQP', bounds=bnds, constraints=cons)
                    opt_w = res.x
                    opt_std, opt_ret = port_perf(opt_w, mean_r, cov_m)
                    opt_shp = (opt_ret - rf_rate) / opt_std
                    
                    st.success("Ottimizzato!")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rendimento Atteso", f"{opt_ret*100:.2f}%")
                    m2.metric("Volatilità", f"{opt_std*100:.2f}%")
                    m3.metric("Sharpe Ratio", f"{opt_shp:.2f}")
                    
                    c_pie, c_hm = st.columns(2)
                    with c_pie:
                        lbls = [valid[i] for i in range(num) if opt_w[i]>0.01]
                        vals = [opt_w[i] for i in range(num) if opt_w[i]>0.01]
                        st.plotly_chart(go.Figure(data=[go.Pie(labels=lbls, values=vals, hole=.4)]), use_container_width=True)
                    with c_hm:
                        st.plotly_chart(go.Figure(data=go.Heatmap(z=rets.corr().values, x=valid, y=valid, colorscale='RdBu', zmin=-1, zmax=1)), use_container_width=True)
                    
                    st.subheader("Frontiera Efficiente")
                    n_sim = 2000
                    w_all = np.zeros((n_sim, num))
                    r_arr = np.zeros(n_sim)
                    v_arr = np.zeros(n_sim)
                    s_arr = np.zeros(n_sim)
                    
                    for i in range(n_sim):
                        w = np.random.random(num); w /= np.sum(w)
                        w_all[i,:] = w
                        v_arr[i], r_arr[i] = port_perf(w, mean_r, cov_m)
                        s_arr[i] = (r_arr[i] - rf_rate) / v_arr[i]
                        
                    ef = go.Figure()
                    ef.add_trace(go.Scatter(x=v_arr, y=r_arr, mode='markers', marker=dict(color=s_arr, colorscale='Viridis', showscale=True), name='Simulazioni'))
                    ef.add_trace(go.Scatter(x=[opt_std], y=[opt_ret], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='OTTIMO'))
                    st.plotly_chart(ef, use_container_width=True)
