import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import feedparser
import scipy.optimize as sco # NUOVO: Per l'ottimizzazione matematica
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. CONFIGURAZIONE ---
st.set_page_config(page_title="STX Ultimate Suite", page_icon="🏦", layout="wide") # Layout Wide per il portafoglio

# --- 2. CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        html, body, [class*="css"], .stTextInput, div, span, p {
            font-family: 'JetBrains Mono', monospace !important;
        }
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
    </style>
""", unsafe_allow_html=True)

# --- 3. NAVIGAZIONE (SIDEBAR) ---
st.sidebar.markdown("## 🕹️ Modalità Operativa")
app_mode = st.sidebar.radio(
    "Scegli lo strumento:",
    ["🔎 Analisi Singola (AI + Sentiment)", "⚖️ Ottimizzatore Portafoglio (Markowitz)"]
)

st.sidebar.markdown("---")
st.sidebar.info("STX Ultimate v4.0\nRunning on Python + Tensorflow + Scipy")

# ==============================================================================
# MODULO 1: ANALISI SINGOLA (IL TUO VECCHIO CODICE POTENZIATO)
# ==============================================================================

if app_mode == "🔎 Analisi Singola (AI + Sentiment)":
    
    st.markdown('<p class="big-title">STX DEEP DIVE</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI + Macro + Monte Carlo + Google News Sentiment</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input("Inserisci Ticker", placeholder="Es. STLA.MI, TSLA, BTC-USD").upper().strip()
    with col2:
        benchmark_input = st.text_input("Benchmark", value="^GSPC", help="Es. ^GSPC, FTSEMIB.MI").upper().strip()

    # --- FUNZIONI DI SUPPORTO ANALISI SINGOLA ---
    def analyze_news_sentiment(ticker):
        try:
            clean_ticker = ticker.split('.')[0]
            rss_url = f"https://news.google.com/rss/search?q={clean_ticker}+stock+market&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            if not feed.entries: return 0, []
            
            analyzer = SentimentIntensityAnalyzer()
            total_score = 0
            analyzed_news = []
            panic_words = ["war", "bankrupt", "fraud", "crash", "crisis", "plunge", "collapse"]
            hype_words = ["soar", "record", "breakthrough", "skyrocket", "surge", "beats"]

            for entry in feed.entries[:10]:
                title = entry.title
                link = entry.link
                publisher = entry.source.title if 'source' in entry else "Google News"
                vs = analyzer.polarity_scores(title)
                score = vs['compound']
                
                title_lower = title.lower()
                if any(w in title_lower for w in panic_words):
                    if score > -0.5: score = -0.6
                elif any(w in title_lower for w in hype_words):
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
                market_cum = df_log['General_Market'].cumsum()
                stock_cum = df_log['Stock_Price'].cumsum()
                rel_strength = float(stock_cum.iloc[-1] - market_cum.iloc[-1])
            except: rel_strength = 0.0
            
            corr = df_log.iloc[-500:].corr()['Stock_Price'].drop(['Stock_Price', 'Stock_Vol'])
            return df, df_log, corr, rel_strength
        except: return None, None, None, None

    @st.cache_resource(show_spinner=False)
    def train_lstm_model(df_log):
        feature_cols = ['Stock_Price', 'Fear_Index', 'Gold_War', 'Oil_Energy', 'Rates_Inflation', 'General_Market']
        existing = [c for c in feature_cols if c in df_log.columns]
        data = np.clip(df_log[existing].values, -0.1, 0.1)
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(data)
        
        X, y = [], []
        PRED_DAYS = 90
        for i in range(PRED_DAYS, len(scaled)):
            X.append(scaled[i-PRED_DAYS:i])
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
        return model, scaler, scaled, existing

    # --- ESECUZIONE ANALISI SINGOLA ---
    if ticker_input:
        progress = st.progress(0, text="Analisi News & Dati...")
        sent_score, news = analyze_news_sentiment(ticker_input)
        df_prices, df_log, corr, rs = get_single_data(ticker_input, benchmark_input)
        
        if df_prices is None:
            st.error("Dati insufficienti. Controlla il Ticker.")
            progress.empty()
        else:
            # 1. SENTIMENT
            st.markdown(f"##### 📰 Sentiment (Google News)")
            if sent_score > 0.2: s_lbl, s_col, s_imp = "BULLISH", "#00ff00", 1.05
            elif sent_score > 0.05: s_lbl, s_col, s_imp = "POSITIVO", "#90ee90", 1.02
            elif sent_score < -0.2: s_lbl, s_col, s_imp = "BEARISH", "#ff0000", 0.95
            elif sent_score < -0.05: s_lbl, s_col, s_imp = "NEGATIVO", "#ff4444", 0.98
            else: s_lbl, s_col, s_imp = "NEUTRALE", "gray", 1.00
            
            c1, c2 = st.columns([1, 2])
            with c1: st.markdown(f"<div style='text-align:center; border:2px solid {s_col}; padding:10px; border-radius:10px;'><div style='font-size:3rem;'>{sent_score:.2f}</div><div style='color:{s_col}; font-weight:bold;'>{s_lbl}</div></div>", unsafe_allow_html=True)
            with c2:
                if news: st.markdown(f"**Top:** [{news[0]['title']}]({news[0]['link']})"); st.caption(f"Fonte: {news[0]['publisher']}")
                else: st.info("Nessuna news recente.")

            # 2. AI MODEL
            progress.progress(40, text="AI Thinking...")
            model, scaler, scaled, cols = train_lstm_model(df_log)
            
            # 3. PROJECTION
            progress.progress(70, text="Simulazione...")
            last_seq = scaled[-90:]
            curr_batch = last_seq.reshape((1, 90, len(cols)))
            fut_ret = []
            macro_trend = np.mean(scaled[-30:, 1:], axis=0) if len(cols) > 1 else []
            
            FUTURE_DAYS = 365
            for i in range(FUTURE_DAYS):
                pred = model.predict(curr_batch, verbose=0)[0, 0]
                pred = np.clip(pred, -0.05, 0.05) * (0.99 ** i)
                fut_ret.append(pred)
                
                new_row = [pred]
                if len(macro_trend) > 0:
                    current_macro = macro_trend * (0.95 ** i) + np.random.normal(0, 0.01, size=len(macro_trend))
                    new_row = np.insert(current_macro, 0, pred)
                
                curr_batch = np.append(curr_batch[:, 1:, :], [new_row], axis=1) # Fix shape mismatch logic if needed
                
            dummy = np.zeros((len(fut_ret), len(cols)))
            dummy[:, 0] = fut_ret
            real_ret = scaler.inverse_transform(dummy)[:, 0]
            
            curr_p = df_prices['Stock_Price'].iloc[-1]
            fut_prices = []
            drift = (s_imp - 1.0) / 30
            for i, r in enumerate(real_ret):
                d = drift if i < 30 else 0
                curr_p *= np.exp(r + d)
                fut_prices.append(curr_p)
                
            dates = [df_prices.index[-1] + datetime.timedelta(days=i) for i in range(1, FUTURE_DAYS+1)]
            
            progress.progress(100, "Fatto.")
            progress.empty()
            
            # CHART
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prices.index[-365:], y=df_prices['Stock_Price'].iloc[-365:], name='Storico', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=dates, y=fut_prices, name='Forecast AI', line=dict(color='#0055ff', width=3)))
            st.plotly_chart(fig, use_container_width=True)
            
            chg = ((fut_prices[-1] - df_prices['Stock_Price'].iloc[-1]) / df_prices['Stock_Price'].iloc[-1]) * 100
            st.markdown(f"<div style='text-align:center;'>Target 1Y: <b>{fut_prices[-1]:.2f}</b> | Trend: <b style='color:{'#00ff00' if chg>0 else '#ff4444'}'>{chg:+.2f}%</b></div>", unsafe_allow_html=True)
            
            # TABS
            t1, t2, t3 = st.tabs(["🧠 Macro", "🔮 Monte Carlo", "🔗 Correlazioni"])
            with t1:
                if corr is not None: st.bar_chart(corr)
            with t2:
                u, v = df_log['Stock_Price'].mean(), df_log['Stock_Price'].var()
                drift, stdev = u - (0.5*v), df_log['Stock_Price'].std()
                days = np.exp(drift + stdev * np.random.normal(0, 1, (365, 1000)))
                paths = np.zeros_like(days); paths[0] = df_prices['Stock_Price'].iloc[-1]
                for t in range(1, 365): paths[t] = paths[t-1] * days[t]
                
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Scatter(x=dates, y=np.percentile(paths, 95, axis=1), name='Best Case', line=dict(color='green')))
                fig_mc.add_trace(go.Scatter(x=dates, y=np.percentile(paths, 5, axis=1), name='Worst Case', line=dict(color='red')))
                fig_mc.add_trace(go.Scatter(x=dates, y=np.percentile(paths, 50, axis=1), name='Median', line=dict(color='white', dash='dot')))
                st.plotly_chart(fig_mc, use_container_width=True)
            with t3:
                try:
                    bt = benchmark_input if benchmark_input else "^GSPC"
                    b_d = yf.Ticker(bt).history(period="max")['Close'].pct_change()
                    s_d = df_prices['Stock_Price'].pct_change()
                    comb = pd.DataFrame({'S': s_d, 'B': b_d}).dropna()
                    roll = comb['S'].rolling(60).corr(comb['B']).dropna()
                    st.line_chart(roll)
                except: st.info("No correlation data.")

# ==============================================================================
# MODULO 2: PORTFOLIO OPTIMIZER (NUOVO - ALADDIN STYLE)
# ==============================================================================

elif app_mode == "⚖️ Ottimizzatore Portafoglio (Markowitz)":
    
    st.markdown('<p class="big-title">PORTFOLIO OPTIMIZER</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Efficient Frontier • Sharpe Ratio Maximization • Asset Allocation</p>', unsafe_allow_html=True)

    # Input multiplo
    default_tickers = "AAPL, MSFT, GOOG, TSLA, STLA.MI, ENI.MI, BTC-USD, GLD"
    tickers_string = st.text_area("Inserisci i Ticker del tuo portafoglio (separati da virgola)", default_tickers, height=70)
    
    col_btn, col_risk = st.columns([1, 2])
    with col_btn:
        run_opt = st.button("🚀 Ottimizza Allocazione", type="primary")
    with col_risk:
        risk_free_rate = st.number_input("Tasso Risk-Free (es. Bond USA 10Y)", value=0.04, step=0.01, format="%.2f")

    # Logica Matematica
    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return std, returns

    def neg_sharpe(weights, mean_returns, cov_matrix, rf):
        p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(p_ret - rf) / p_var

    if run_opt:
        tickers_list = [x.strip().upper() for x in tickers_string.split(',') if x.strip()]
        
        if len(tickers_list) < 2:
            st.error("Inserisci almeno 2 asset per creare un portafoglio.")
        else:
            with st.spinner("Scaricamento dati e calcolo correlazioni..."):
                # DOWNLOAD BULLETPROOF PER LISTE MISTE (USA + EU)
                closes = pd.DataFrame()
                valid_tickers = []
                
                for t in tickers_list:
                    try:
                        # Scarica dati storici (2 anni)
                        data = yf.Ticker(t).history(period="2y")
                        if not data.empty:
                            # Fix Timezone
                            data.index = pd.to_datetime(data.index).tz_localize(None)
                            # Usa Close o Adj Close
                            price = data['Close']
                            closes[t] = price
                            valid_tickers.append(t)
                    except:
                        st.warning(f"Impossibile scaricare {t}")

                if len(valid_tickers) < 2:
                    st.error("Non ci sono abbastanza dati validi per ottimizzare.")
                else:
                    # Pulizia e Calcoli
                    closes = closes.ffill().bfill().dropna()
                    returns = closes.pct_change()
                    mean_returns = returns.mean()
                    cov_matrix = returns.cov()
                    num_assets = len(valid_tickers)

                    # Ottimizzazione con Scipy
                    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
                    initial_guess = num_assets * [1. / num_assets,]

                    opt_results = sco.minimize(
                        neg_sharpe, 
                        initial_guess, 
                        args=(mean_returns, cov_matrix, risk_free_rate), 
                        method='SLSQP', 
                        bounds=bounds, 
                        constraints=constraints
                    )
                    
                    opt_weights = opt_results.x
                    opt_vol, opt_ret = portfolio_performance(opt_weights, mean_returns, cov_matrix)
                    opt_sharpe = (opt_ret - risk_free_rate) / opt_vol

                    # --- VISUALIZZAZIONE RISULTATI ---
                    st.success("Ottimizzazione Completata!")
                    
                    # 1. Metriche
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rendimento Atteso (Annuo)", f"{opt_ret*100:.2f}%", help="Quanto ti aspetti di guadagnare mediamente in un anno.")
                    m2.metric("Volatilità (Rischio)", f"{opt_vol*100:.2f}%", help="Quanto oscilla il portafoglio. Più basso è meglio.")
                    m3.metric("Sharpe Ratio", f"{opt_sharpe:.2f}", help="Efficienza: Rendimento per unità di rischio. >1 è buono, >2 è ottimo.")

                    # 2. Grafici
                    c_pie, c_corr = st.columns(2)
                    
                    with c_pie:
                        st.subheader("🍰 Allocazione Ideale")
                        # Filtra asset con peso < 1% per pulizia
                        labels = [valid_tickers[i] for i in range(num_assets) if opt_weights[i] > 0.01]
                        values = [opt_weights[i] for i in range(num_assets) if opt_weights[i] > 0.01]
                        
                        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
                        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=350)
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with c_corr:
                        st.subheader("🔥 Heatmap Correlazioni")
                        fig_hm = go.Figure(data=go.Heatmap(
                            z=returns.corr().values,
                            x=valid_tickers,
                            y=valid_tickers,
                            colorscale='RdBu', zmin=-1, zmax=1
                        ))
                        fig_hm.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig_hm, use_container_width=True)
                        st.caption("Rosso = Si muovono insieme (Rischioso). Blu = Si muovono opposti (Diversificazione).")

                    # 3. Frontiera Efficiente (Bonus Visivo)
                    st.subheader("📈 Frontiera Efficiente (Risk vs Return)")
                    
                    # Genera portafogli casuali per il contesto
                    n_portfolios = 2000
                    all_weights = np.zeros((n_portfolios, num_assets))
                    ret_arr = np.zeros(n_portfolios)
                    vol_arr = np.zeros(n_portfolios)
                    sharpe_arr = np.zeros(n_portfolios)

                    for i in range(n_portfolios):
                        w = np.random.random(num_assets)
                        w /= np.sum(w)
                        all_weights[i, :] = w
                        vol_arr[i], ret_arr[i] = portfolio_performance(w, mean_returns, cov_matrix)
                        sharpe_arr[i] = (ret_arr[i] - risk_free_rate) / vol_arr[i]

                    fig_ef = go.Figure()
                    
                    # Nuvola di punti
                    fig_ef.add_trace(go.Scatter(
                        x=vol_arr, y=ret_arr, mode='markers',
                        marker=dict(color=sharpe_arr, colorscale='Viridis', showscale=True, size=5, colorbar=dict(title="Sharpe")),
                        name='Portafogli Casuali'
                    ))
                    
                    # Stella Ottimale
                    fig_ef.add_trace(go.Scatter(
                        x=[opt_vol], y=[opt_ret], mode='markers',
                        marker=dict(color='red', size=15, symbol='star'),
                        name='PORTAFOGLIO OTTIMO'
                    ))
                    
                    fig_ef.update_layout(
                        xaxis_title="Volatilità (Rischio)", 
                        yaxis_title="Rendimento Atteso",
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(128,128,128,0.1)',
                        height=500
                    )
                    st.plotly_chart(fig_ef, use_container_width=True)
