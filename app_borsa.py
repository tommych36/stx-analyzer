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

# --- 1. CONFIGURAZIONE ---
st.set_page_config(page_title="STX Ultimate Stable", page_icon="🛡️", layout="centered")

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
st.markdown('<p class="subtitle">Macro-Economics + Deep History + <b>Safety Logic</b></p>', unsafe_allow_html=True)

ticker_input = st.text_input(
    "Inserisci Ticker", 
    placeholder="Es. LDO.MI, TSLA, BTC-USD, ENI.MI", 
    help="Analisi correllata con Oro, Petrolio, Tassi e Volatilità."
).upper().strip()

# --- 4. MOTORE IBRIDO STABILIZZATO ---
PREDICTION_DAYS = 90    
FUTURE_DAYS = 365       

@st.cache_data(ttl=12*3600)
def get_ultimate_data(ticker):
    try:
        # Scarica MAX dati
        stock = yf.download(ticker, period="max", interval="1d", progress=False)
        if len(stock) < 300: return None, None, None, None

        # Scarica Macro
        tickers = ["^VIX", "GC=F", "CL=F", "^TNX", "^GSPC"]
        macro_data = yf.download(tickers, period="max", interval="1d", progress=False)['Close']
        
        if isinstance(stock.columns, pd.MultiIndex): stock.columns = stock.columns.get_level_values(0)
        
        df = stock[['Close']].rename(columns={'Close': 'Stock_Price'})
        df = df.join(macro_data, how='inner')
        
        df.rename(columns={
            '^VIX': 'Fear_Index', 'GC=F': 'Gold_War', 'CL=F': 'Oil_Energy', 
            '^TNX': 'Rates_Inflation', '^GSPC': 'General_Market'
        }, inplace=True)

        # RENDIMENTI LOGARITMICI (Più sicuri per l'IA)
        df_log = np.log(df / df.shift(1)).fillna(0)
        
        # Volatilità (sui prezzi reali)
        df_log['Stock_Vol'] = df['Stock_Price'].pct_change().rolling(20).std().fillna(0)
        
        # Forza Relativa
        try:
            market_cum = df_log['General_Market'].cumsum()
            stock_cum = df_log['Stock_Price'].cumsum()
            val = stock_cum.iloc[-1] - market_cum.iloc[-1]
            relative_strength = float(val)
        except:
            relative_strength = 0.0

        # Correlazioni
        recent_corr = df_log.iloc[-500:].corr()['Stock_Price'].drop(['Stock_Price', 'Stock_Vol'])

        return df, df_log, recent_corr, relative_strength

    except Exception as e:
        return None, None, None, None

@st.cache_resource(show_spinner=False)
def train_ultimate_model(df_log):
    # Training su Log Returns
    feature_cols = ['Stock_Price', 'Fear_Index', 'Gold_War', 'Oil_Energy', 'Rates_Inflation', 'General_Market']
    data_values = df_log[feature_cols].values

    # Clip estremo per i dati di training (toglie i giorni assurdi tipo +20%)
    # Questo insegna all'IA a ignorare i crolli/boom anomali
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
    progress_bar = st.progress(0, text="Analisi Macro e Dati Storici...")
    
    # Nota: df_log contiene le variazioni, df_prices i prezzi reali
    df_prices, df_log, correlations, rel_strength = get_ultimate_data(ticker_input)
    
    if df_prices is None:
        st.error("Dati insufficienti.")
        progress_bar.empty()
    else:
        # 1. MOSTRA CORRELAZIONI
        st.markdown("##### 🧠 Macro-Brain: Cosa muove il prezzo?")
        if correlations is not None:
            corr_fig = go.Figure(go.Bar(
                x=correlations.index, y=correlations.values,
                marker_color=['#ff4444' if x < 0 else '#00ff41' for x in correlations.values]
            ))
            corr_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=250, margin=dict(l=10,r=10,t=10,b=20),
                font=dict(family="JetBrains Mono"),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(corr_fig, use_container_width=True)

        # 2. TRAINING
        progress_bar.progress(40, text="Training Neurale (Log-Returns)...")
        model, scaler, scaled_data, feature_cols = train_ultimate_model(df_log)
        
        # 3. PROIEZIONE STABILIZZATA
        progress_bar.progress(70, text="Simulazione Scenari (Safety ON)...")
        
        last_sequence = scaled_data[-PREDICTION_DAYS:]
        current_batch = last_sequence.reshape((1, PREDICTION_DAYS, len(feature_cols)))
        future_log_returns = []
        
        # Trend Macro Inerziale (ma che tende a zero nel tempo)
        recent_macro_trend = np.mean(scaled_data[-30:, 1:], axis=0) 
        
        for i in range(FUTURE_DAYS):
            pred_log_ret = model.predict(current_batch, verbose=0)[0, 0]
            
            # --- BLOCCO DI SICUREZZA (SAFETY CLAMPS) ---
            
            # 1. CLIPPING: Impediamo movimenti > 5% al giorno (in scala normalizzata)
            # Questo uccide i picchi assurdi che causano l'esplosione
            pred_log_ret = np.clip(pred_log_ret, -0.05, 0.05)
            
            # 2. DECAY AGGRESSIVO: Più andiamo avanti, più il trend si spegne
            # Questo curva il grafico invece di farlo andare dritto all'infinito
            decay = 0.99 ** i 
            pred_log_ret *= decay
            
            future_log_returns.append(pred_log_ret)
            
            # 3. MACRO RETURN TO MEAN: Anche i fattori esterni (oro, tassi) tornano alla media
            # Non assumiamo che l'oro salga per sempre
            current_macro = recent_macro_trend * (0.95 ** i) # Decade verso 0
            
            # Aggiungiamo rumore per realismo
            noise = np.random.normal(0, 0.01, size=len(current_macro))
            new_macro_values = current_macro + noise
            
            new_row = np.insert(new_macro_values, 0, pred_log_ret)
            current_batch = np.append(current_batch[:, 1:, :], [[new_row]], axis=1)

        # Inversione
        dummy_matrix = np.zeros((len(future_log_returns), len(feature_cols)))
        dummy_matrix[:, 0] = future_log_returns
        future_real_log_returns = scaler.inverse_transform(dummy_matrix)[:, 0]

        # Ricostruzione Prezzo (da Log Return a Prezzo)
        last_price = df_prices['Stock_Price'].iloc[-1]
        future_prices = []
        curr_p = last_price
        
        for ret in future_real_log_returns:
            # Formula inversa del Log Return
            curr_p = curr_p * np.exp(ret)
            future_prices.append(curr_p)
            
        last_date = df_prices.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, FUTURE_DAYS + 1)]

        progress_bar.progress(100, text="Fatto.")
        progress_bar.empty()

        # 4. GRAFICO
        fig = go.Figure()
        
        # Storico
        past = df_prices.iloc[-365:]
        fig.add_trace(go.Scatter(x=past.index, y=past['Stock_Price'], mode='lines', name='Storico', line=dict(color='var(--text-color)', width=2)))
        
        # Zone Volatilità
        # Calcoliamo soglia volatilità sui dati recenti
        vol_data = df_log['Stock_Vol'].iloc[-365:]
        high_vol = vol_data.quantile(0.95)
        
        fig.add_trace(go.Scatter(
            x=past.index,
            y=[past['Stock_Price'].max() if v > high_vol else None for v in vol_data],
            fill='tozeroy', fillcolor='rgba(255, 50, 50, 0.15)', mode='none', name='Alta Volatilità', hoverinfo='skip'
        ))

        # Forecast
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Forecast AI', line=dict(color='#0055ff', width=3)))
        
        fig.add_vline(x=last_date, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=f"SCENARIO ULTIMATE: {ticker_input}",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(128,128,128,0.05)',
            font=dict(family="JetBrains Mono"), height=550
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 5. DATI FINALI
        chg = ((future_prices[-1] - last_price) / last_price) * 100
        
        if rel_strength > 0: rs_color, rs_sign = "#00ff00", "+"
        else: rs_color, rs_sign = "#ff4444", ""
        
        st.markdown(f"""
        <div style="text-align:center; font-size:1.1rem;">
            Target 1Y: <b>{future_prices[-1]:.2f}</b> | Trend: <b style='color:{'#00ff00' if chg>0 else '#ff4444'}'>{chg:+.2f}%</b> <br>
            <span style="font-size:0.9rem; color:gray">Forza Relativa vs S&P500: <b style="color:{rs_color}">{rs_sign}{rel_strength*100:.2f}%</b></span>
        </div>
        """, unsafe_allow_html=True)
