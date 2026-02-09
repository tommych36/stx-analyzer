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

# --- 2. CSS "GHOST MODE" & STILE ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        
        html, body, [class*="css"], .stTextInput, div, span, p {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* --- FIX SIDEBAR TOGGLE --- */
        [data-testid="stSidebarCollapsedControl"] {
            visibility: hidden !important;
            width: 50px !important; 
            height: 50px !important;
            position: relative;
        }

        [data-testid="stSidebarCollapsedControl"]::after {
            content: "➤"; 
            visibility: visible !important;
            font-size: 24px !important;
            color: #808080;
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            display: flex; align-items: center; justify_content: center;
            transition: color 0.3s;
        }
        
        [data-testid="stSidebarCollapsedControl"]:hover::after {
            color: #000000; 
            cursor: pointer;
        }

        [data-testid="stSidebar"] button[kind="header"]::after {
            content: "◀"; 
            visibility: visible !important;
            font-size: 24px !important;
            color: #808080;
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            display: flex; align-items: center; justify_content: center;
        }

        .big-title { 
            text-align: center; 
            font-size: 3rem !important; 
            font-weight: 700; 
            margin-bottom: 10px; 
            color: var(--text-color); 
        }
        .subtitle { 
            text-align: center; 
            font-size: 1rem; 
            color: gray; 
            margin-bottom: 30px; 
        }
        
        /* Box Esplicativi */
        .explanation-box {
            background-color: #f0f2f6; 
            border-left: 5px solid #0055ff;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
            margin-bottom: 25px;
            font-size: 0.95rem;
            color: #31333F;
        }
        
        /* Stile News List */
        .news-item {
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .news-meta {
            font-size: 0.8rem;
            color: #666;
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
st.sidebar.info("STX Ultimate v5.5\nPass Score UI")

# ==============================================================================
# MODALITÀ 1: ANALISI SINGOLA (DEEP DIVE)
# ==============================================================================
if app_mode == "🔎 Analisi Singola (Deep Dive)":
    
    st.markdown('<p class="big-title">STX DEEP DIVE</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI + Macro + Monte Carlo + Google & Yahoo News</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1: 
        ticker_input = st.text_input("Inserisci Ticker", placeholder="Es. STLA.MI, TSLA, BTC-USD").upper().strip()
    with col2: 
        benchmark_input = st.text_input("Benchmark", value="^GSPC").upper().strip()

    # --- NEWS SENTIMENT (MULTI-SOURCE) ---
    def analyze_news_sentiment(ticker):
        try:
            clean_ticker = ticker.split('.')[0]
            
            # 1. Lista fonti RSS
            rss_urls = [
                f"https://news.google.com/rss/search?q={clean_ticker}+stock+market&hl=en-US&gl=US&ceid=US:en",
                f"https://finance.yahoo.com/rss/headline?s={clean_ticker}"
            ]
            
            all_entries = []
            
            # 2. Scaricamento
            for url in rss_urls:
                try:
                    feed = feedparser.parse(url)
                    if feed.entries:
                        all_entries.extend(feed.entries)
                except: continue 

            if not all_entries: return 0, []
            
            analyzer = SentimentIntensityAnalyzer()
            total_score = 0
            analyzed_news = []
            seen_titles = set()
            
            panic_words = ["war", "bankrupt", "fraud", "crash", "crisis", "collapse"]
            hype_words = ["soar", "record", "skyrocket", "surge", "buy", "beats"]

            # 3. Analisi
            for entry in all_entries[:60]: 
                title = entry.title
                if title in seen_titles: continue
                seen_titles.add(title)

                link = entry.link
                if 'source' in entry: publisher = entry.source.title 
                elif 'yfinance' in link or 'yahoo' in link: publisher = "Yahoo Finance"
                else: publisher = "News Source"
                
                vs = analyzer.polarity_scores(title)
                score = vs['compound']
                
                if any(w in title.lower() for w in panic_words) and score > -0.5: score = -0.6
                if any(w in title.lower() for w in hype_words) and score < 0.5: score = 0.6
                
                total_score += score
                
                if score >= 0.05: color = "#00ff00"
                elif score <= -0.05: color = "#ff4444"
                else: color = "gray"
                
                analyzed_news.append({'title': title, 'link': link, 'score': score, 'color': color, 'publisher': publisher})
                
            if not analyzed_news: return 0, []
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

    # --- LSTM MODEL ---
    @st.cache_resource(show_spinner=False)
    def train_lstm(df_log):
        cols = ['Stock_Price', 'Fear_Index', 'Gold_War', 'Oil_Energy', 'Rates_Inflation', 'General_Market']
        exist = [c for c in cols if c in df_log.columns]
        data = np.clip(df_log[exist].values, -0.1, 0.1)
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(data)
        
        X, y = [], []
        PD = 90
        for i in range(PD, len(scaled)):
            X.append(scaled[i-PD:i])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        
        model = Sequential()
        model.add(Input(shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=15, batch_size=128, verbose=0)
        return model, scaler, scaled, exist

    # --- ESECUZIONE ---
    if ticker_input:
        progress = st.progress(0, "Analisi in corso...")
        s_score, news = analyze_news_sentiment(ticker_input)
        df_p, df_l, corr, rs = get_single_data(ticker_input, benchmark_input)
        
        if df_p is None:
            st.error("Dati insufficienti."); progress.empty()
        else:
            # --- SEZIONE NEWS UI (RATING 0-10 + PASS SCORE) ---
            st.markdown("##### 📰 News Sentiment Analysis")
            
            # Formula: ((Score + 1) / 2) * 10
            vote_display = ((s_score + 1) / 2) * 10
            
            if s_score > 0.2: l, c, imp = "BULLISH", "#00ff00", 1.05
            elif s_score > 0.05: l, c, imp = "POSITIVO", "#90ee90", 1.02
            elif s_score < -0.2: l, c, imp = "BEARISH", "#ff0000", 0.95
            elif s_score < -0.05: l, c, imp = "NEGATIVO", "#ff4444", 0.98
            else: l, c, imp = "NEUTRALE", "gray", 1.00
            
            c1, c2 = st.columns([1, 2])
            
            # Score Gigante + Pass Score Text
            with c1: 
                st.markdown(f"<div style='text-align:center; border:2px solid {c}; padding:10px; border-radius:10px; margin-bottom:5px;'><div style='font-size:3rem;'>{vote_display:.1f}<span style='font-size:1.5rem; color:gray;'>/10</span></div><div style='color:{c}; font-weight:bold;'>{l}</div></div>", unsafe_allow_html=True)
                st.markdown("<div style='text-align:center; color:gray; font-size:0.8rem;'>Pass Score: 5/10</div>", unsafe_allow_html=True)
            
            # Lista Classificata
            with c2:
                if news:
                    st.caption("🔥 Top Articoli (Voto d'Impatto 0-10):")
                    sorted_news = sorted(news, key=lambda x: abs(x['score']), reverse=True)
                    
                    with st.container(height=200):
                        for n in sorted_news:
                            item_vote = ((n['score'] + 1) / 2) * 10
                            
                            if n['score'] >= 0.05: icon = "🟢"
                            elif n['score'] <= -0.05: icon = "🔴"
                            else: icon = "⚪"
                            
                            st.markdown(f"""
                            <div class="news-item">
                                <div>{icon} <a href="{n['link']}" target="_blank" style="text-decoration:none; color:inherit; font-weight:bold;">{n['title']}</a></div>
                                <div class="news-meta">
                                    Fonte: {n['publisher']} | Voto: <b>{item_vote:.1f}/10</b>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("Nessuna news recente trovata.")
            # -----------------------------------------

            # AI Training
            progress.progress(40, "AI Training...")
            model, scaler, scaled, cols = train_lstm(df_l)
            
            # Simulation
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
                
                if len(macro_trend) > 0:
                    mac = macro_trend * (0.95**i) + np.random.normal(0, 0.01, len(macro_trend))
                    new_row = np.insert(mac, 0, p)
                else:
                    new_row = np.array([p])
                
                new_row = new_row.reshape(1, 1, len(new_row)) 
                curr = np.append(curr[:, 1:, :], new_row, axis=1)
                
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
            
            # Main Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p.index[-365:], y=df_p['Stock_Price'].iloc[-365:], name='Storico', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=dates, y=fut_p, name='Forecast AI', line=dict(color='#0055ff', width=3)))
            st.plotly_chart(fig, use_container_width=True)
            
            # --- EDUCATIONAL TEXT: PREVISIONE ---
            final_p = fut_p[-1]
            start_p = df_p['Stock_Price'].iloc[-1]
            perc_chg = ((final_p - start_p) / start_p) * 100
            
            if perc_chg > 2: trend_desc = "POSITIVO (Rialzista)"
            elif perc_chg < -2: trend_desc = "NEGATIVO (Ribassista)"
            else: trend_desc = "NEUTRO (Laterale)"
            
            news_factor = "supporta" if (s_score > 0 and perc_chg > 0) or (s_score < 0 and perc_chg < 0) else "contrasta"
            
            st.markdown(f"""
            <div class="explanation-box">
                <b>📊 Interpretazione Previsione:</b><br>
                L'Intelligenza Artificiale stima che tra un anno il prezzo potrebbe essere <b>{final_p:.2f}</b>, indicando un trend <b>{trend_desc}</b>.<br>
                <br>
                <b>Perché?</b><br>
                1. <b>Analisi Tecnica AI:</b> Il modello ha rilevato pattern storici che suggeriscono questa direzione.<br>
                2. <b>Analisi Fondamentale News:</b> Il sentiment attuale delle notizie (Voto: {vote_display:.1f}/10) <b>{news_factor}</b> questa previsione.
            </div>
            """, unsafe_allow_html=True)
            
            # Stats text
            chg = ((fut_p[-1] - df_p['Stock_Price'].iloc[-1]) / df_p['Stock_Price'].iloc[-1])*100
            rc = "#00ff00" if rs > 0 else "#ff4444"
            rsign = "+" if rs > 0 else ""
            st.markdown(f"<div style='text-align:center; font-size:1.1rem;'>Target 1Y: <b>{fut_p[-1]:.2f}</b> | Trend: <b style='color:{'#00ff00' if chg>0 else '#ff4444'}'>{chg:+.2f}%</b><br><span style='color:gray; font-size:0.9rem;'>Forza Relativa: <b style='color:{rc}'>{rsign}{rs*100:.2f}%</b></span></div>", unsafe_allow_html=True)
            
            # Tabs
            t1, t2, t3 = st.tabs(["🧠 Macro", "🔮 Monte Carlo", "🔗 Correlazioni"])
            
            with t1: 
                f_corr = go.Figure(go.Bar(x=corr.index, y=corr.values, marker_color=['red' if x<0 else 'green' for x in corr.values]))
                st.plotly_chart(f_corr, use_container_width=True)
                
                # --- EDUCATIONAL TEXT: MACRO ---
                expl_macro = []
                if 'Fear_Index' in corr.index:
                    rel = "INVERSA" if corr['Fear_Index'] < 0 else "DIRETTA"
                    meaning = "il titolo tende a SCENDERE quando nel mercato c'è paura (VIX alto)." if corr['Fear_Index'] < 0 else "il titolo sale insieme alla paura (asset rifugio?)."
                    expl_macro.append(f"• <b>Paura (VIX):</b> Relazione {rel}. Storicamente {meaning}")
                    
                if 'Gold_War' in corr.index:
                     rel = "DIRETTA" if corr['Gold_War'] > 0 else "INVERSA"
                     meaning = "il titolo segue l'andamento dell'oro (bene rifugio)." if corr['Gold_War'] > 0 else "il titolo soffre quando i capitali si spostano sull'oro."
                     expl_macro.append(f"• <b>Oro & Geopolitica:</b> Relazione {rel}. {meaning}")
                
                if 'Rates_Inflation' in corr.index:
                    rel = "INVERSA" if corr['Rates_Inflation'] < 0 else "DIRETTA"
                    meaning = "il titolo soffre l'aumento dei tassi d'interesse." if corr['Rates_Inflation'] < 0 else "il titolo beneficia dei tassi alti (es. bancari)."
                    expl_macro.append(f"• <b>Tassi d'Interesse:</b> Relazione {rel}. {meaning}")

                macro_text = "<br>".join(expl_macro)
                st.markdown(f"""
                <div class="explanation-box">
                    <b>🧠 Spiegazione Macroeconomica:</b><br>
                    Questo grafico ti dice "chi comanda" il prezzo dell'azione oltre alla speculazione.<br>
                    <br>
                    {macro_text}
                </div>
                """, unsafe_allow_html=True)
            
            with t2:
                u, v = df_l['Stock_Price'].mean(), df_l['Stock_Price'].var()
                dr, sd = u-(0.5*v), df_l['Stock_Price'].std()
                days = np.exp(dr + sd * np.random.normal(0, 1, (FD, 1000)))
                paths = np.zeros_like(days); paths[0] = df_p['Stock_Price'].iloc[-1]
                for t in range(1, FD): paths[t] = paths[t-1] * days[t]
                
                fmc = go.Figure()
                fmc.add_trace(go.Scatter(x=dates, y=np.percentile(paths, 95, axis=1), name='Best', line=dict(color='green')))
                fmc.add_trace(go.Scatter(x=dates, y=np.percentile(paths, 5, axis=1), name='Worst', line=dict(color='red')))
                fmc.add_trace(go.Scatter(x=dates, y=np.percentile(paths, 50, axis=1), name='Median', line=dict(color='black', dash='dot')))
                st.plotly_chart(fmc, use_container_width=True)
                
                loss = ((df_p['Stock_Price'].iloc[-1] - np.percentile(paths, 5, axis=1)[-1]) / df_p['Stock_Price'].iloc[-1])*100
                st.error(f"⚠️ Value at Risk (95%): Nello scenario peggiore statistico, rischio max: -{loss:.2f}%")
                
                # --- EDUCATIONAL TEXT: MONTE CARLO ---
                st.markdown(f"""
                <div class="explanation-box">
                    <b>🎲 Interpretazione del Rischio (Monte Carlo):</b><br>
                    Abbiamo simulato 1000 universi paralleli per il prossimo anno.<br>
                    <ul>
                    <li><b>Linea Verde (Best Case):</b> Se tutto va benissimo (top 5% degli scenari).</li>
                    <li><b>Linea Rossa (Worst Case):</b> Se tutto va male (peggior 5% degli scenari).</li>
                    <li><b>Value at Risk (VaR):</b> Quel numero rosso sopra indica il rischio statistico. Esempio: se è -20%, significa che c'è il 95% di probabilità che NON perderai più del 20%.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with t3:
                try:
                    bt = benchmark_input if benchmark_input else "^GSPC"
                    bd = yf.Ticker(bt).history(period="max")
                    bd.index = pd.to_datetime(bd.index).tz_localize(None)
                    comb = pd.DataFrame({'A': df_p['Stock_Price'].pct_change(), 'B': bd['Close'].pct_change()}).dropna()
                    roll = comb['A'].rolling(60).corr(comb['B']).dropna()
                    st.line_chart(roll)
                    
                    # --- EDUCATIONAL TEXT: CORRELAZIONI ---
                    last_c = roll.iloc[-1]
                    if last_c > 0.7: c_type = "MOLTO FORTE"
                    elif last_c > 0.3: c_type = "MODERATA"
                    elif last_c > -0.3: c_type = "NULLA (Decorrelato)"
                    else: c_type = "INVERSA (Copertura)"
                    
                    st.markdown(f"""
                    <div class="explanation-box">
                        <b>🔗 Guida alle Correlazioni:</b><br>
                        Questo grafico mostra quanto il titolo "copia" i movimenti del mercato ({bt}).<br>
                        <ul>
                        <li><b>+1.0 (Max):</b> Si muovono identici.</li>
                        <li><b>0.0 (Zero):</b> Il titolo ignora il mercato.</li>
                        <li><b>-1.0 (Opposto):</b> Se il mercato sale, il titolo scende.</li>
                        </ul>
                        Attualmente la correlazione è <b>{last_c:.2f}</b>, quindi il legame è <b>{c_type}</b>.
                    </div>
                    """, unsafe_allow_html=True)
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
    with c_risk: 
        rf_rate = st.number_input("Risk-Free Rate", 0.04, step=0.01)
        # --- EDUCATIONAL TEXT: RISK FREE ---
        st.caption("ℹ️ Inserisci il rendimento di un titolo di stato sicuro (es. BTP o Treasury a 10 anni). Serve come base per calcolare quanto 'rischio' vale la pena correre.")

    def port_perf(weights, mean_ret, cov):
        ret = np.sum(mean_ret * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
        return std, ret

    def neg_sharpe(weights, mean_ret, cov, rf):
        std, ret = port_perf(weights, mean_ret, cov)
        return -(ret - rf) / std

    if run_opt:
        t_list = [x.strip().upper() for x in tickers_str.split(',') if x.strip()]
        if len(t_list) < 2: st.error("Servono almeno 2 asset.")
        else:
            with st.spinner("Scaricamento e Ottimizzazione..."):
                closes = pd.DataFrame()
                valid_tickers = []
                
                for t in t_list:
                    try:
                        d = yf.Ticker(t).history(period="2y")
                        if not d.empty:
                            d.index = pd.to_datetime(d.index).tz_localize(None)
                            closes[t] = d['Close']
                            valid_tickers.append(t)
                    except: st.warning(f"Errore su {t}")
                
                if len(valid_tickers) < 2: st.error("Dati insufficienti.")
                else:
                    closes = closes.ffill().bfill().dropna()
                    rets = closes.pct_change()
                    mean_r = rets.mean()
                    cov_m = rets.cov()
                    num = len(valid_tickers)
                    
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
                        lbls = [valid_tickers[i] for i in range(num) if opt_w[i]>0.01]
                        vals = [opt_w[i] for i in range(num) if opt_w[i]>0.01]
                        st.plotly_chart(go.Figure(data=[go.Pie(labels=lbls, values=vals, hole=.4)]), use_container_width=True)
                    with c_hm:
                        st.plotly_chart(go.Figure(data=go.Heatmap(z=rets.corr().values, x=valid_tickers, y=valid_tickers, colorscale='RdBu', zmin=-1, zmax=1)), use_container_width=True)
                    
                    # --- FRONTIERA EFFICIENTE ---
                    st.subheader("Frontiera Efficiente (Simulazione)")
                    n_sim = 2000
                    w_all = np.zeros((n_sim, num))
                    r_arr = np.zeros(n_sim)
                    v_arr = np.zeros(n_sim)
                    s_arr = np.zeros(n_sim)
                    
                    for i in range(n_sim):
                        w = np.random.random(num)
                        w /= np.sum(w)
                        w_all[i,:] = w
                        v_arr[i], r_arr[i] = port_perf(w, mean_r, cov_m)
                        s_arr[i] = (r_arr[i] - rf_rate) / v_arr[i]
                        
                    ef = go.Figure()
                    ef.add_trace(go.Scatter(x=v_arr, y=r_arr, mode='markers', marker=dict(color=s_arr, colorscale='Viridis', showscale=True), name='Simulazioni'))
                    ef.add_trace(go.Scatter(x=[opt_std], y=[opt_ret], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='OTTIMO'))
                    st.plotly_chart(ef, use_container_width=True)
