import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
# Assicurati di aver installato: pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. CONFIGURAZIONE ---
st.set_page_config(page_title="STX Ultimate Full", page_icon="🛡️", layout="centered")

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
        .stTextInput > div > div > input {
            text-align: center; border-radius: 12px; padding: 12px; 
            border: 2px solid var(--text-color); font-weight: bold;
        }
        .stPlotlyChart {
            background-color: var(--secondary-background-color); 
            border-radius: 15px; padding: 10px; margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. INTERFACCIA ---
st.markdown('<p class="big-title">STX ULTIMATE</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI + Macro + Monte Carlo + News Sentiment Analysis</p>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    ticker_input = st.text_input(
        "Inserisci Ticker", 
        placeholder="Es. STLA.MI, TSLA, BTC-USD", 
        help="Analisi su tutto lo storico disponibile."
    ).upper().strip()

with col2:
    benchmark_input = st.text_input(
        "Benchmark", 
        value="^GSPC", 
        help="Asset di confronto (es. ^GSPC, ^IXIC, FTSEMIB.MI)"
    ).upper().strip()

# --- 4. MOTORE SENTIMENT ---
def analyze_news_sentiment(ticker):
    try:
        t = yf.Ticker(ticker)
        news_list = t.news
        if not news_list: return 0, []

        analyzer = SentimentIntensityAnalyzer()
        total_score = 0
        analyzed_news = []
        panic_words = ["war", "bankrupt", "fraud", "crash", "investigation", "crisis"]

        for item in news_list[:10]:
            title = item.get('title', '')
            link = item.get('link', '')
            publisher = item.get('publisher', 'Unknown')
            
            vs = analyzer.polarity_scores(title)
            score = vs['compound']
            
            if any(w in title.lower() for w in panic_words):
                score *= 1.5 
                
            total_score += score
            color = "#00ff00" if score > 0.05 else "#ff4444" if score < -0.05 else "gray"
            
            analyzed_news.append({'title': title, 'link': link, 'score': score, 'color': color, 'publisher': publisher})
            
        avg_sentiment = total_score / len(analyzed_news) if analyzed_news else 0
        return avg_sentiment, analyzed_news
    except:
        return 0, []

# --- 5. MOTORE IBRIDO BLINDATO (NEW v3) ---
PREDICTION_DAYS = 90    
FUTURE_DAYS = 365       

@st.cache_data(ttl=12*3600)
def get_ultimate_data(ticker, benchmark_ticker):
    try:
        # 1. SCARICA STOCK CON METODO "SICURO" (.history)
        # Questo metodo evita i problemi di formattazione MultiIndex di yf.download
        stock_obj = yf.Ticker(ticker)
        stock = stock_obj.history(period="max")
        
        # Se è vuoto, proviamo a scaricare con yf.download come fallback
        if stock.empty:
            stock = yf.download(ticker, period="max", progress=False)
        
        # Se ancora vuoto o troppo corto, usciamo
        if stock is None or len(stock) < 300:
            return None, None, None, None

        # Standardizzazione Indice Temporale
        stock.index = pd.to_datetime(stock.index).tz_localize(None)
        
        # Creiamo il DataFrame principale solo con il prezzo di chiusura
        # Usiamo 'Close' o 'Adj Close' se disponibile
        if 'Close' in stock.columns:
            df = stock[['Close']].copy()
        elif 'Adj Close' in stock.columns:
            df = stock[['Adj Close']].copy()
        else:
            return None, None, None, None # Colonne non trovate

        df.rename(columns={df.columns[0]: 'Stock_Price'}, inplace=True)

        # 2. SCARICA MACRO SINGOLARMENTE (Per non rompere tutto se uno fallisce)
        macro_dict = {
            "^VIX": "Fear_Index", 
            "GC=F": "Gold_War", 
            "CL=F": "Oil_Energy", 
            "^TNX": "Rates_Inflation"
        }
        
        for symbol, name in macro_dict.items():
            try:
                # Scarichiamo solo l'ultimo anno se MAX fallisce, ma qui proviamo MAX
                m_data = yf.Ticker(symbol).history(period="max")
                m_data.index = pd.to_datetime(m_data.index).tz_localize(None)
                # Unione "Left" (mantiene le date dello stock originale)
                df[name] = m_data['Close']
            except:
                df[name] = 0.0 # Se fallisce, riempi con 0

        # 3. SCARICA BENCHMARK
        if not benchmark_ticker: benchmark_ticker = "^GSPC"
        
        try:
            b_data = yf.Ticker(benchmark_ticker).history(period="max")
            if b_data.empty: raise Exception("Empty Benchmark")
            b_data.index = pd.to_datetime(b_data.index).tz_localize(None)
            df['General_Market'] = b_data['Close']
        except:
            # Fallback intelligente: se il benchmark utente fallisce, usa S&P500
            try:
                b_data = yf.Ticker("^GSPC").history(period="max")
                b_data.index = pd.to_datetime(b_data.index).tz_localize(None)
                df['General_Market'] = b_data['Close']
            except:
                df['General_Market'] = df['Stock_Price'] # Fallback estremo

        # 4. PULIZIA DATI (Filling Intelligente)
        # Riempiamo i buchi dei macro (es. festività USA) con il dato del giorno prima
        df = df.ffill().bfill()
        
        # Cancelliamo SOLO se manca il prezzo dello stock
        df.dropna(subset=['Stock_Price'], inplace=True)

        if len(df) < 300: return None, None, None, None

        # 5. CALCOLI FINALI
        df_log = np.log(df / df.shift(1)).fillna(0)
        df_log['Stock_Vol'] = df['Stock_Price'].pct_change().rolling(20).std().fillna(0)
        
        try:
            market_cum = df_log['General_Market'].cumsum()
            stock_cum = df_log['Stock_Price'].cumsum()
            val = stock_cum.iloc[-1] - market_cum.iloc[-1]
            relative_strength = float(val)
        except:
            relative_strength = 0.0

        recent_corr = df_log.iloc[-500:].corr()['Stock_Price'].drop(['Stock_Price', 'Stock_Vol'])

        return df, df_log, recent_corr, relative_strength

    except Exception as e:
        # print(f"DEBUG ERROR: {e}") 
        return None, None, None, None

@st.cache_resource(show_spinner=False)
def train_ultimate_model(df_log):
    feature_cols = ['Stock_Price', 'Fear_Index', 'Gold_War', 'Oil_Energy', 'Rates_Inflation', 'General_Market']
    
    # Verifica che le colonne esistano (in caso di fallback estremi)
    existing_cols = [c for c in feature_cols if c in df_log.columns]
    data_values = df_log[existing_cols].values
    data_values = np.clip(data_values, -0.1, 0.1) 

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data_values)

    x_train, y_train = [], []
    for i in range(PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[i-PREDICTION_DAYS:i]) 
        y_train.append(scaled_data[i, 0]) 

    x_train, y_train = np.array(x_train), np.array(y_train)

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2]))) 
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=15, batch_size=128, verbose=0)
    
    return model, scaler, scaled_data, existing_cols

# --- ESECUZIONE ---
if ticker_input:
    progress_bar = st.progress(0, text="Scansione News & Dati Globali...")
    
    # 1. ANALISI SENTIMENT
    sentiment_score, news_items = analyze_news_sentiment(ticker_input)
    
    # 2. ANALISI DATI
    df_prices, df_log, correlations, rel_strength = get_ultimate_data(ticker_input, benchmark_input)
    
    if df_prices is None:
        st.error(f"Dati insufficienti per {ticker_input}. Potrebbe essere un ticker non valido o delisting.")
        progress_bar.empty()
    else:
        # --- DISPLAY SENTIMENT ---
        st.markdown(f"##### 📰 News Sentiment Analysis (Ultimi articoli)")
        
        if sentiment_score > 0.2: sent_label, sent_color, sentiment_impact = "MOLTO POSITIVO", "#00ff00", 1.05
        elif sentiment_score > 0.05: sent_label, sent_color, sentiment_impact = "POSITIVO", "#90ee90", 1.02
        elif sentiment_score < -0.2: sent_label, sent_color, sentiment_impact = "MOLTO NEGATIVO", "#ff0000", 0.95
        elif sentiment_score < -0.05: sent_label, sent_color, sentiment_impact = "NEGATIVO", "#ff4444", 0.98
        else: sent_label, sent_color, sentiment_impact = "NEUTRALE", "gray", 1.00

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.markdown(f"""
            <div style="text-align:center; border: 2px solid {sent_color}; padding: 10px; border-radius: 10px;">
                <div style="font-size: 3rem;">{sentiment_score:.2f}</div>
                <div style="color: {sent_color}; font-weight: bold;">{sent_label}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s2:
            if news_items:
                top_news = news_items[0]
                st.markdown(f"**Top News:** [{top_news['title']}]({top_news['link']})")
                st.caption(f"Fonte: {top_news['publisher']}")
            else:
                st.info("Nessuna news recente rilevata.")

        # --- TRAINING & PREVISIONE ---
        progress_bar.progress(40, text="Training Neurale...")
        model, scaler, scaled_data, feature_cols = train_ultimate_model(df_log)
        
        progress_bar.progress(70, text="Simulazione Scenari...")
        
        last_sequence = scaled_data[-PREDICTION_DAYS:]
        current_batch = last_sequence.reshape((1, PREDICTION_DAYS, len(feature_cols)))
        future_log_returns = []
        recent_macro_trend = np.mean(scaled_data[-30:, 1:], axis=0) 
        
        for i in range(FUTURE_DAYS):
            pred_log_ret = model.predict(current_batch, verbose=0)[0, 0]
            pred_log_ret = np.clip(pred_log_ret, -0.05, 0.05)
            decay = 0.99 ** i 
            pred_log_ret *= decay
            future_log_returns.append(pred_log_ret)
            
            # Simulazione Macro inerziale
            if len(recent_macro_trend) > 0:
                current_macro = recent_macro_trend * (0.95 ** i)
                noise = np.random.normal(0, 0.01, size=len(current_macro))
                new_macro_values = current_macro + noise
                new_row = np.insert(new_macro_values, 0, pred_log_ret)
            else:
                new_row = [pred_log_ret] # Fallback se mancano colonne macro

            current_batch = np.append(current_batch[:, 1:, :], [[new_row]], axis=1)

        # Inversione
        dummy_matrix = np.zeros((len(future_log_returns), len(feature_cols)))
        dummy_matrix[:, 0] = future_log_returns
        future_real_log_returns = scaler.inverse_transform(dummy_matrix)[:, 0]

        last_price = df_prices['Stock_Price'].iloc[-1]
        future_prices = []
        curr_p = last_price
        
        daily_sentiment_drift = (sentiment_impact - 1.0) / 30 
        
        for i, ret in enumerate(future_real_log_returns):
            extra_drift = daily_sentiment_drift if i < 30 else 0
            curr_p = curr_p * np.exp(ret + extra_drift)
            future_prices.append(curr_p)
            
        last_date = df_prices.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, FUTURE_DAYS + 1)]

        progress_bar.progress(100, text="Fatto.")
        progress_bar.empty()

        # --- GRAFICI ---
        fig = go.Figure()
        past = df_prices.iloc[-365:]
        fig.add_trace(go.Scatter(x=past.index, y=past['Stock_Price'], mode='lines', name='Storico', line=dict(color='var(--text-color)', width=2)))
        
        vol_data = df_log['Stock_Vol'].iloc[-365:]
        high_vol = vol_data.quantile(0.95)
        fig.add_trace(go.Scatter(x=past.index, y=[past['Stock_Price'].max() if v > high_vol else None for v in vol_data], fill='tozeroy', fillcolor='rgba(255, 50, 50, 0.15)', mode='none', name='Alta Volatilità', hoverinfo='skip'))

        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Forecast AI (+Sentiment)', line=dict(color='#0055ff', width=3)))
        fig.add_vline(x=last_date, line_dash="dash", line_color="red")
        
        fig.update_layout(title=f"SCENARIO ULTIMATE: {ticker_input}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(128,128,128,0.05)', font=dict(family="JetBrains Mono"), height=550)
        st.plotly_chart(fig, use_container_width=True)
        
        chg = ((future_prices[-1] - last_price) / last_price) * 100
        rs_color = "#00ff00" if rel_strength > 0 else "#ff4444"
        rs_sign = "+" if rel_strength > 0 else ""
        
        st.markdown(f"""
        <div style="text-align:center; font-size:1.1rem; margin-bottom: 30px;">
            Target 1Y: <b>{future_prices[-1]:.2f}</b> | Trend: <b style='color:{'#00ff00' if chg>0 else '#ff4444'}'>{chg:+.2f}%</b> <br>
            <span style="font-size:0.9rem; color:gray">Forza Relativa vs {benchmark_input}: <b style="color:{rs_color}">{rs_sign}{rel_strength*100:.2f}%</b></span>
        </div>
        """, unsafe_allow_html=True)

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["🧠 Macro Brain", "🔮 Monte Carlo", "🔗 Correlazioni"])

        with tab1:
            if correlations is not None:
                corr_fig = go.Figure(go.Bar(x=correlations.index, y=correlations.values, marker_color=['#ff4444' if x < 0 else '#00ff41' for x in correlations.values]))
                corr_fig.update_layout(title="Correlazioni Fattori Globali", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(corr_fig, use_container_width=True)

        with tab2:
            log_returns = df_log['Stock_Price'] 
            u, var = log_returns.mean(), log_returns.var()
            drift, stdev = u - (0.5 * var), log_returns.std()
            daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (FUTURE_DAYS, 1000)))
            
            S0 = df_prices['Stock_Price'].iloc[-1]
            price_list = np.zeros_like(daily_returns)
            price_list[0] = S0
            for t in range(1, FUTURE_DAYS): price_list[t] = price_list[t - 1] * daily_returns[t]

            q05, q50, q95 = np.percentile(price_list, 5, axis=1), np.percentile(price_list, 50, axis=1), np.percentile(price_list, 95, axis=1)
            
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=future_dates, y=q95, mode='lines', line=dict(color='rgba(0,255,0,0.5)', width=1), name='Best Case'))
            fig_mc.add_trace(go.Scatter(x=future_dates, y=q05, mode='lines', line=dict(color='rgba(255,0,0,0.5)', width=1), fill='tonexty', name='Worst Case'))
            fig_mc.add_trace(go.Scatter(x=future_dates, y=q50, mode='lines', line=dict(color='white', width=2, dash='dot'), name='Median'))
            fig_mc.update_layout(title="Monte Carlo (1000 Scenari)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(128,128,128,0.05)', height=400)
            st.plotly_chart(fig_mc, use_container_width=True)
            
            loss_pct = ((S0 - q05[-1]) / S0) * 100
            st.error(f"⚠️ Value at Risk (95%): Rischio massimo statistico stimato: -{loss_pct:.2f}%")

        with tab3:
            try:
                # Metodo sicuro anche per il Benchmark
                bench_ticker_clean = benchmark_input if benchmark_input else "^GSPC"
                b_data = yf.Ticker(bench_ticker_clean).history(period="max")
                b_data.index = pd.to_datetime(b_data.index).tz_localize(None)
                
                combined = pd.DataFrame({'Asset': df_prices['Stock_Price'].pct_change(), 'Bench': b_data['Close'].pct_change()}).dropna()
                roll_corr = combined['Asset'].rolling(60).corr(combined['Bench']).dropna()
                
                if not roll_corr.empty:
                    fig_corr = go.Figure()
                    fig_corr.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr.values, mode='lines', line=dict(color='#ff00ff', width=2)))
                    fig_corr.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=0.8, x1=1, y1=1.0, fillcolor="rgba(255,0,0,0.1)", line_width=0)
                    fig_corr.update_layout(title=f"Rolling Correlation vs {bench_ticker_clean}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Dati insufficienti per correlazione.")
            except:
                st.info("Impossibile caricare grafico correlazione.")
