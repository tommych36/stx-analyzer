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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # NUOVO CERVELLO LINGUISTICO

# --- 1. CONFIGURAZIONE ---
st.set_page_config(page_title="STX Ultimate Sentiment", page_icon="🧠", layout="centered")

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
        .news-box {
            border: 1px solid rgba(128,128,128,0.2); border-radius: 10px; 
            padding: 15px; margin-bottom: 10px; background-color: rgba(255,255,255,0.02);
        }
        .sentiment-score { font-size: 1.2rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 3. INTERFACCIA ---
st.markdown('<p class="big-title">STX ULTIMATE</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI + Macro + Monte Carlo + <b>News Sentiment Analysis</b></p>', unsafe_allow_html=True)

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
        help="Asset di confronto (es. ^GSPC, ^IXIC, BTC-USD)"
    ).upper().strip()

# --- 4. MOTORE SENTIMENT (NUOVO) ---
def analyze_news_sentiment(ticker):
    """
    Scarica le news recenti da Yahoo Finance e calcola il sentiment medio.
    Restituisce: Score (-1 a 1), Lista News Rilevanti
    """
    try:
        t = yf.Ticker(ticker)
        news_list = t.news
        
        if not news_list:
            return 0, []

        analyzer = SentimentIntensityAnalyzer()
        total_score = 0
        analyzed_news = []
        
        # Parole chiave "Killer" (Eventi gravi che devono pesare di più)
        panic_words = ["war", "bankrupt", "fraud", "crash", "investigation", "crisis"]

        for item in news_list[:10]: # Analizza le ultime 10 news
            title = item.get('title', '')
            link = item.get('link', '')
            publisher = item.get('publisher', 'Unknown')
            
            # Analisi VADER
            vs = analyzer.polarity_scores(title)
            score = vs['compound']
            
            # Bonus/Malus per parole chiave gravi
            if any(w in title.lower() for w in panic_words):
                score *= 1.5 # Amplifica l'impatto negativo/positivo
                
            total_score += score
            
            # Determina colore per display
            if score > 0.05: color = "#00ff00"
            elif score < -0.05: color = "#ff4444"
            else: color = "gray"
            
            analyzed_news.append({
                'title': title,
                'link': link,
                'score': score,
                'color': color,
                'publisher': publisher
            })
            
        avg_sentiment = total_score / len(analyzed_news) if analyzed_news else 0
        return avg_sentiment, analyzed_news

    except Exception as e:
        return 0, []

# --- 5. MOTORE IBRIDO STABILIZZATO ---
PREDICTION_DAYS = 90    
FUTURE_DAYS = 365       

@st.cache_data(ttl=12*3600)
def get_ultimate_data(ticker, benchmark_ticker):
    try:
        stock = yf.download(ticker, period="max", interval="1d", progress=False)
        if len(stock) < 300: return None, None, None, None

        tickers = ["^VIX", "GC=F", "CL=F", "^TNX", benchmark_ticker]
        macro_data = yf.download(tickers, period="max", interval="1d", progress=False)['Close']
        
        if isinstance(stock.columns, pd.MultiIndex): stock.columns = stock.columns.get_level_values(0)
        
        stock.index = stock.index.tz_localize(None)
        macro_data.index = macro_data.index.tz_localize(None)
        
        df = stock[['Close']].rename(columns={'Close': 'Stock_Price'})
        df = df.join(macro_data, how='left').ffill().bfill()
        
        df.rename(columns={
            '^VIX': 'Fear_Index', 'GC=F': 'Gold_War', 'CL=F': 'Oil_Energy', 
            '^TNX': 'Rates_Inflation', benchmark_ticker: 'General_Market'
        }, inplace=True)
        
        df.dropna(inplace=True)
        if len(df) < 300: return None, None, None, None

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
        return None, None, None, None

@st.cache_resource(show_spinner=False)
def train_ultimate_model(df_log):
    feature_cols = ['Stock_Price', 'Fear_Index', 'Gold_War', 'Oil_Energy', 'Rates_Inflation', 'General_Market']
    data_values = df_log[feature_cols].values
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
    
    return model, scaler, scaled_data, feature_cols

# --- ESECUZIONE ---
if ticker_input:
    progress_bar = st.progress(0, text="Scansione News & Dati Globali...")
    
    # 1. ANALISI SENTIMENT (News)
    sentiment_score, news_items = analyze_news_sentiment(ticker_input)
    
    # 2. ANALISI DATI
    df_prices, df_log, correlations, rel_strength = get_ultimate_data(ticker_input, benchmark_input)
    
    if df_prices is None:
        st.error(f"Dati insufficienti per {ticker_input}.")
        progress_bar.empty()
    else:
        # --- DISPLAY SENTIMENT ---
        st.markdown(f"##### 📰 News Sentiment Analysis (Ultimi articoli)")
        
        # Interpretazione Score
        if sentiment_score > 0.2: 
            sent_label, sent_color = "MOLTO POSITIVO (Bullish)", "#00ff00"
            sentiment_impact = 1.05 # +5% boost al target
        elif sentiment_score > 0.05: 
            sent_label, sent_color = "POSITIVO", "#90ee90"
            sentiment_impact = 1.02
        elif sentiment_score < -0.2: 
            sent_label, sent_color = "MOLTO NEGATIVO (Bearish)", "#ff0000"
            sentiment_impact = 0.95 # -5% taglio al target
        elif sentiment_score < -0.05: 
            sent_label, sent_color = "NEGATIVO", "#ff4444"
            sentiment_impact = 0.98
        else: 
            sent_label, sent_color = "NEUTRALE", "gray"
            sentiment_impact = 1.00

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
                st.caption(f"Fonte: {top_news['publisher']} | VADER Score: {top_news['score']:.2f}")
                st.info("L'IA correggerà la previsione matematica in base a questo sentiment.")
            else:
                st.warning("Nessuna news recente trovata. L'analisi sarà puramente tecnica.")

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
            
            current_macro = recent_macro_trend * (0.95 ** i)
            noise = np.random.normal(0, 0.01, size=len(current_macro))
            new_macro_values = current_macro + noise
            new_row = np.insert(new_macro_values, 0, pred_log_ret)
            current_batch = np.append(current_batch[:, 1:, :], [[new_row]], axis=1)

        dummy_matrix = np.zeros((len(future_log_returns), len(feature_cols)))
        dummy_matrix[:, 0] = future_log_returns
        future_real_log_returns = scaler.inverse_transform(dummy_matrix)[:, 0]

        last_price = df_prices['Stock_Price'].iloc[-1]
        future_prices = []
        curr_p = last_price
        
        # APPLICAZIONE SENTIMENT ALLA PREVISIONE
        # Distribuiamo l'impatto del sentiment gradualmente nei primi 30 giorni
        daily_sentiment_drift = (sentiment_impact - 1.0) / 30 
        
        for i, ret in enumerate(future_real_log_returns):
            # Aggiungiamo il "sentiment drift" solo per il primo mese
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

        # --- SEZIONE MACRO & RISK (TABS) ---
        tab1, tab2, tab3 = st.tabs(["🧠 Macro Brain", "🔮 Monte Carlo", "🔗 Correlazioni"])

        with tab1:
            if correlations is not None:
                corr_fig = go.Figure(go.Bar(x=correlations.index, y=correlations.values, marker_color=['#ff4444' if x < 0 else '#00ff41' for x in correlations.values]))
                corr_fig.update_layout(title="Correlazioni Fattori Globali", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=10,r=10,t=40,b=20))
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
            if benchmark_input:
                try:
                    bench_data = yf.download(benchmark_input, period="max", interval="1d", progress=False)
                    if isinstance(bench_data.columns, pd.MultiIndex): bench_data.columns = bench_data.columns.get_level_values(0)
                    bench_data.index = bench_data.index.tz_localize(None)
                    
                    combined = pd.DataFrame({'Asset': df_prices['Stock_Price'].pct_change(), 'Bench': bench_data['Close'].pct_change()}).dropna()
                    roll_corr = combined['Asset'].rolling(60).corr(combined['Bench']).dropna()
                    
                    fig_corr = go.Figure()
                    fig_corr.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr.values, mode='lines', line=dict(color='#ff00ff', width=2)))
                    fig_corr.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=0.8, x1=1, y1=1.0, fillcolor="rgba(255,0,0,0.1)", line_width=0)
                    fig_corr.update_layout(title=f"Rolling Correlation vs {benchmark_input}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
                    st.plotly_chart(fig_corr, use_container_width=True)
                except: st.error("Errore benchmark.")
