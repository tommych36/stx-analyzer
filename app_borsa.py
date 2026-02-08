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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="STX Analyzer Pro", page_icon="📈", layout="centered")

# --- 2. CSS ADATTIVO (DARK/LIGHT MODE + FONT JETBRAINS) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        
        html, body, [class*="css"], .stTextInput, .stMarkdown, .stPlotlyChart, div, span, p, .stSelectbox {
            font-family: 'JetBrains Mono', monospace !important;
        }

        .big-title {
            text-align: center; font-size: 3rem !important; font-weight: 700;
            margin-bottom: 20px; letter-spacing: -2px;
            color: var(--text-color);
        }

        /* Input e Menu */
        .stTextInput > div > div > input, .stSelectbox > div > div {
            text-align: center; border-radius: 50px; padding: 10px; 
            border: 1px solid var(--text-color);
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        
        /* Tasto Freccia */
        .stButton > button {
            border-radius: 50%; width: 50px; height: 50px;
            border: 1px solid var(--text-color);
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            font-size: 20px; transition: all 0.3s ease;
        }
        .stButton > button:hover {
            border-color: #00ff41; color: #00ff41; transform: scale(1.1);
        }

        /* Grafico */
        .stPlotlyChart {
            background-color: var(--secondary-background-color); 
            border-radius: 20px; padding: 15px;
            box-shadow: 0px 10px 30px rgba(0,0,0,0.2); margin-top: 40px;
        }

        #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. LISTA STOCK E UI ---
@st.cache_data
def get_ticker_list():
    custom_list = [
        "STLA.MI - Stellantis (MI)", "RACE.MI - Ferrari (MI)", 
        "RNO.PA - Renault", "ISP.MI - Intesa Sanpaolo", "UCG.MI - Unicredit", 
        "ENI.MI - Eni", "LDO.MI - Leonardo", "TIT.MI - Telecom Italia", 
        "BTC-USD - Bitcoin", "ETH-USD - Ethereum"
    ]
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500_list = (sp500['Symbol'] + " - " + sp500['Security']).tolist()
        final_list = custom_list + sp500_list
    except:
        final_list = custom_list + ["AAPL - Apple", "MSFT - Microsoft", "TSLA - Tesla", "NVDA - Nvidia"]
    return sorted(final_list)

st.markdown('<p class="big-title">STX ANALYZER</p>', unsafe_allow_html=True)
col1, col2 = st.columns([6, 1])
with col1:
    selected_item = st.selectbox("Seleziona Stock", options=get_ticker_list(), index=None, placeholder="Cerca (es. STLA...)", label_visibility="collapsed")
with col2:
    search_pressed = st.button("➔")

target_ticker = selected_item.split(" - ")[0] if selected_item else None
start_analysis = True if target_ticker and (search_pressed or selected_item) else False

# --- 4. IL MOTORE "COMPLESSO" (Logica Colab Originale) ---
PREDICTION_DAYS = 90    # Memoria a breve termine (3 mesi)
FUTURE_DAYS = 365       # PREVISIONE A 1 ANNO (365 Giorni)

def add_fundamental_proxies(df):
    # Logica esatta del Colab
    df['Volatility'] = df['Close'].rolling(window=10).std()
    vol_ma = df['Volume'].rolling(window=50).mean()
    df['News_Impact'] = df['Volume'] / vol_ma
    
    # Indicatori Tecnici (pandas_ta)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd.iloc[:, 0] # Fix per nomi colonne dinamici
    
    df.dropna(inplace=True)
    return df

if start_analysis:
    progress_text = f"Analisi Profonda (1 Anno) su {target_ticker}..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # 1. Download Esteso
        my_bar.progress(10, text="Download storico completo...")
        data = yf.download(target_ticker, period="max", interval="1d", progress=False)
        
        if len(data) < 365: # Serve più storico per il modello complesso
            st.error(f"Dati insufficienti per {target_ticker}. Serve almeno 1 anno di storico.")
            my_bar.empty()
        else:
            # 2. Feature Engineering
            my_bar.progress(20, text="Calcolo Volatilità, RSI, MACD...")
            df = data.copy()
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = add_fundamental_proxies(df)
            
            feature_columns = ['Close', 'Volatility', 'News_Impact', 'RSI', 'MACD']
            dataset = df[feature_columns].values
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            x_train, y_train = [], []
            for i in range(PREDICTION_DAYS, len(scaled_data)):
                x_train.append(scaled_data[i-PREDICTION_DAYS:i]) 
                y_train.append(scaled_data[i, 0]) 

            x_train, y_train = np.array(x_train), np.array(y_train)

            # 3. Modello "Pesante" (Bi-LSTM Deep)
            my_bar.progress(40, text="Costruzione Architettura Neurale Bidirezionale...")
            model = Sequential()
            model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
            
            # Architettura identica al Colab
            model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
            model.add(Dropout(0.3))
            
            model.add(Dense(units=50, activation='relu'))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='huber') # Uso 'huber' invece di 'huber_loss' per compatibilità

            # 4. Addestramento
            my_bar.progress(50, text="Addestramento in corso (Pazienta, calcoli complessi)...")
            # Riduco leggermente le epoche per il Cloud gratuito, ma mantengo la logica
            callbacks = [
                EarlyStopping(monitor='loss', patience=6, restore_best_weights=True),
                ReduceLROnPlateau(monitor='loss', patience=3, factor=0.5)
            ]
            model.fit(x_train, y_train, epochs=15, batch_size=64, callbacks=callbacks, verbose=0)
            
            # 5. Proiezione a 365 Giorni
            my_bar.progress(80, text=f"Generazione scenario futuro ({FUTURE_DAYS} giorni)...")
            
            last_sequence = scaled_data[-PREDICTION_DAYS:]
            current_batch = last_sequence.reshape((1, PREDICTION_DAYS, len(feature_columns)))
            
            future_predictions = []
            recent_volatility = np.std(df['Close'].iloc[-30:]) / df['Close'].iloc[-1]

            for i in range(FUTURE_DAYS):
                pred_price_scaled = model.predict(current_batch, verbose=0)[0, 0]
                # Micro-rumore per realismo
                noise = np.random.normal(0, recent_volatility * 0.05) 
                pred_price_scaled += noise
                
                future_predictions.append(pred_price_scaled)
                
                new_row = current_batch[0, -1, :].copy() 
                new_row[0] = pred_price_scaled
                current_batch = np.append(current_batch[:, 1:, :], [[new_row]], axis=1)

            dummy_matrix = np.zeros((len(future_predictions), len(feature_columns)))
            dummy_matrix[:, 0] = future_predictions
            future_prices = scaler.inverse_transform(dummy_matrix)[:, 0]

            last_date = df.index[-1]
            future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, FUTURE_DAYS + 1)]

            my_bar.progress(100, text="Analisi Completata.")
            my_bar.empty()

            # --- 6. VISUALIZZAZIONE "STILE MATPLOTLIB" IN PLOTLY ---
            fig = go.Figure()

            # A. Passato (Ultimo anno) - Linea Nera/Grigia
            past_subset = df.iloc[-365:]
            fig.add_trace(go.Scatter(
                x=past_subset.index, y=past_subset['Close'], 
                mode='lines', name='Storico (1 Anno)',
                line=dict(color='var(--text-color)', width=2) # Colore adattivo
            ))

            # B. Zone Rosse (Volatilità) - Simulazione fill_between
            high_vol_threshold = past_subset['Volatility'].quantile(0.90)
            # Creiamo rettangoli rossi dove la volatilità è alta
            high_vol_dates = past_subset[past_subset['Volatility'] > high_vol_threshold].index
            
            # Aggiungiamo le barre rosse di sfondo
            if len(high_vol_dates) > 0:
                # Disegniamo i rettangoli usando le shapes di Plotly
                # Per non appesantire, usiamo una logica semplificata: Scatter rosso trasparente
                fig.add_trace(go.Scatter(
                    x=past_subset.index,
                    y=[past_subset['Close'].max() if v > high_vol_threshold else None for v in past_subset['Volatility']],
                    mode='lines',
                    line=dict(width=0),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.1)', # Rosso trasparente
                    hoverinfo='skip',
                    name='Alta Volatilità'
                ))

            # C. Futuro (1 Anno) - Linea Blu Elettrico
            fig.add_trace(go.Scatter(
                x=future_dates, y=future_prices, 
                mode='lines', name='Previsione AI (1 Anno)',
                line=dict(color='#0000FF', width=2.5) # Blu puro come nel grafico Matplotlib
            ))

            # D. Linea Verticale "OGGI" (Rossa Tratteggiata)
            fig.add_vline(x=last_date, line_width=2, line_dash="dash", line_color="red")

            # Layout Pulito ma Simile a Matplotlib
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(255,255,255,0.05)', # Leggero sfondo griglia
                font=dict(family="JetBrains Mono"),
                xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', title="Data"), 
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', title="Prezzo"),
                showlegend=True, margin=dict(l=20,r=20,t=40,b=20), height=600,
                title=dict(text=f"SCENARIO {target_ticker}: Passato vs Futuro (1 Anno)", x=0.5)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Dati Numerici
            curr = df['Close'].iloc[-1]
            pred = future_prices[-1]
            diff = ((pred - curr) / curr) * 100
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: -10px; font-size: 1rem;">
                Current: <b>{curr:.2f}€</b> &nbsp;|&nbsp; 
                Target (1 Anno): <b>{pred:.2f}€</b> &nbsp;|&nbsp; 
                Trend: <b style="color: {'green' if diff > 0 else 'red'}">{diff:+.2f}%</b>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Errore tecnico: {e}")
