import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="STX Analyzer Pro", page_icon="🚀", layout="centered")

# --- 2. CSS STILE JETBRAINS & COLORS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        
        html, body, [class*="css"], .stTextInput, .stMarkdown, .stPlotlyChart, div, span, p {
            font-family: 'JetBrains Mono', monospace !important;
        }
        .big-title {
            text-align: center; font-size: 3rem !important; font-weight: 700;
            margin-bottom: 20px; letter-spacing: -2px; color: var(--text-color);
        }
        /* Input Box Stile "Terminal" */
        .stTextInput > div > div > input {
            text-align: center; border-radius: 10px; padding: 12px; 
            border: 2px solid var(--text-color); 
            background-color: var(--secondary-background-color);
            color: var(--text-color); font-weight: bold;
        }
        /* Grafico */
        .stPlotlyChart {
            background-color: var(--secondary-background-color); 
            border-radius: 15px; padding: 10px;
            box-shadow: 0px 10px 30px rgba(0,0,0,0.1); margin-top: 20px;
        }
        #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. INTERFACCIA DI RICERCA LIBERA ---
st.markdown('<p class="big-title">STX ANALYZER</p>', unsafe_allow_html=True)

# Nessuna lista precaricata. Libertà totale.
ticker_input = st.text_input(
    "Inserisci Ticker (Mondiale)", 
    placeholder="Scrivi qui... (es. TSLA, RACE.MI, BTC-USD, LVMH.PA)", 
    help="Puoi cercare qualsiasi azione esistente su Yahoo Finance."
).upper().strip() # Rende tutto maiuscolo e toglie spazi

# --- 4. MOTORE AI "REALISTA" (Returns based) ---
PREDICTION_DAYS = 60    # Guarda gli ultimi 2 mesi
FUTURE_DAYS = 365       # Prevedi 1 anno

@st.cache_data(ttl=12*3600)
def get_data_with_indicators(ticker):
    try:
        # Scarichiamo più dati per avere stabilità
        data = yf.download(ticker, period="5y", interval="1d", progress=False)
        if len(data) < 300: return None
        
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # --- FIX PESSIMISMO: Lavoriamo sui RENDIMENTI (Returns), non sui PREZZI ---
        # L'AI impara "di quanto sale/scende in %", così non ha paura dei massimi storici
        df['Return'] = df['Close'].pct_change()
        
        # Indicatori Tecnici (sui prezzi originali per le zone rosse)
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df.dropna(inplace=True) # Rimuovi i primi giorni vuoti
        return df
    except:
        return None

@st.cache_resource
def train_model(ticker_name, returns_data):
    # Scaliamo i rendimenti (solitamente tra -0.10 e +0.10)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(returns_data.values.reshape(-1, 1))

    x_train, y_train = [], []
    for i in range(PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[i-PREDICTION_DAYS:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Modello LSTM Ottimizzato per Trend
    model = Sequential()
    # Meno neuroni ma più focalizzati per evitare "overfitting" sul rumore
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1)) # Predice il rendimento di domani

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    return model, scaler, scaled_data

# --- ESECUZIONE ---
if ticker_input:
    progress_bar = st.progress(0, text="Avvio Analisi...")
    
    df = get_data_with_indicators(ticker_input)
    
    if df is None:
        st.error(f"Ticker '{ticker_input}' non trovato o dati insufficienti. Riprova (es. prova 'AAPL').")
        progress_bar.empty()
    else:
        progress_bar.progress(30, text="Analisi Volatilità e Trend...")
        
        # Addestriamo SOLO sui Rendimenti ('Return')
        model, scaler, scaled_data = train_model(ticker_input, df['Return'])
        
        progress_bar.progress(60, text="Generazione Scenario Futuro...")
        
        # --- PROIEZIONE FUTURA ---
        # Partiamo dagli ultimi dati reali
        last_sequence = scaled_data[-PREDICTION_DAYS:]
        current_batch = last_sequence.reshape((1, PREDICTION_DAYS, 1))
        
        future_returns = []
        
        # Simulazione
        for i in range(FUTURE_DAYS):
            pred_return_scaled = model.predict(current_batch, verbose=0)[0, 0]
            future_returns.append(pred_return_scaled)
            
            # Aggiorna il batch per il giorno dopo
            current_batch = np.append(current_batch[:, 1:, :], [[[pred_return_scaled]]], axis=1)

        # Convertiamo i rendimenti previsti scalati in rendimenti reali
        future_returns = scaler.inverse_transform(np.array(future_returns).reshape(-1, 1))
        
        # --- RICOSTRUZIONE PREZZO (Dal rendimento al Prezzo Reale) ---
        last_real_price = df['Close'].iloc[-1]
        future_prices = []
        current_price = last_real_price
        
        # Aggiungiamo un leggero "Drift" di mercato (crescita naturale) per evitare stagnazione eccessiva
        # Le azioni storicamente salgono dello 0.03% al giorno in media
        market_drift = 0.0003 

        for ret in future_returns:
            # Prezzo Domani = Prezzo Oggi * (1 + Rendimento Previsto + Drift)
            next_price = current_price * (1 + ret[0] + market_drift)
            future_prices.append(next_price)
            current_price = next_price

        # Date Future
        last_date = df.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, FUTURE_DAYS + 1)]

        progress_bar.progress(100, text="Grafico Pronto!")
        progress_bar.empty()

        # --- GRAFICO FINALE ---
        fig = go.Figure()

        # 1. STORICO (Nero/Grigio)
        past_subset = df.iloc[-365:] # Mostra solo ultimo anno
        fig.add_trace(go.Scatter(
            x=past_subset.index, y=past_subset['Close'],
            mode='lines', name='Storico (1 Anno)',
            line=dict(color='var(--text-color)', width=2)
        ))

        # 2. ZONE ROSSE (Alta Volatilità)
        high_vol_threshold = past_subset['Volatility'].quantile(0.90)
        # Creiamo un'area rossa unica usando il fill
        fig.add_trace(go.Scatter(
            x=past_subset.index,
            y=[past_subset['Close'].max() if v > high_vol_threshold else None for v in past_subset['Volatility']],
            fill='tozeroy', fillcolor='rgba(255, 50, 50, 0.15)', # Rosso leggero
            mode='none', name='Alta Volatilità', hoverinfo='skip'
        ))

        # 3. PREVISIONE (Blu Elettrico)
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            mode='lines', name='Forecast AI',
            line=dict(color='#0044ff', width=3) # Blu Tesla style
        ))

        # Linea OGGI
        fig.add_vline(x=last_date, line_width=2, line_dash="dash", line_color="red")

        fig.update_layout(
            title=dict(text=f"SCENARIO: {ticker_input} (1 Anno)", x=0.5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(128,128,128,0.05)',
            font=dict(family="JetBrains Mono"),
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            margin=dict(l=20,r=20,t=40,b=20), height=600,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistiche sotto
        final_price = future_prices[-1]
        perc_change = ((final_price - last_real_price) / last_real_price) * 100
        
        st.markdown(f"""
        <div style="text-align: center; font-size: 1.1rem; margin-top: -10px;">
            Current: <b>{last_real_price:.2f}€/$</b> &nbsp;|&nbsp; 
            Target (1Y): <b>{final_price:.2f}€/$</b> &nbsp;|&nbsp; 
            Trend: <b style="color: {'#00cc00' if perc_change > 0 else '#ff3333'}">{perc_change:+.2f}%</b>
        </div>
        """, unsafe_allow_html=True)
