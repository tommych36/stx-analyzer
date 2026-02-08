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
st.set_page_config(page_title="STX Analyzer", page_icon="📈", layout="centered")

# --- 2. CSS ADATTIVO (DARK/LIGHT MODE AUTOMATICO) ---
# Usiamo le variabili var(--...) così Streamlit cambia i colori da solo in base al tuo telefono
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        
        /* Font globale */
        html, body, [class*="css"], .stTextInput, .stMarkdown, .stPlotlyChart, div, span, p, .stSelectbox {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Titolo che si adatta al tema (non forziamo il nero) */
        .big-title {
            text-align: center; font-size: 3rem !important; font-weight: 700;
            margin-bottom: 20px; letter-spacing: -2px;
            color: var(--text-color); /* Colore automatico */
        }

        /* Stile Input e Menu: Sfondo semi-trasparente per funzionare su Dark e Light */
        .stTextInput > div > div > input, .stSelectbox > div > div {
            text-align: center; 
            font-size: 1.2rem; 
            border-radius: 50px;
            padding: 10px; 
            border: 1px solid var(--text-color); /* Bordo del colore del testo */
            background-color: var(--secondary-background-color); /* Sfondo adattivo */
            color: var(--text-color);
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        
        /* Pulsante Freccia */
        .stButton > button {
            border-radius: 50%;
            width: 50px;
            height: 50px;
            border: 1px solid var(--text-color);
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            font-size: 20px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            border-color: #00ff41;
            color: #00ff41;
            transform: scale(1.1);
        }

        /* Grafico Galleggiante Adattivo */
        .stPlotlyChart {
            background-color: var(--secondary-background-color); 
            border-radius: 20px; 
            padding: 15px;
            box-shadow: 0px 10px 30px rgba(0,0,0,0.2); 
            margin-top: 40px;
        }

        /* Nascondi elementi inutili */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. LISTA STOCK INTELLIGENTE ---
@st.cache_data
def get_ticker_list():
    custom_list = [
        "RACE.MI - Ferrari (Milano)", "STLA.MI - Stellantis (Milano)", 
        "ISP.MI - Intesa Sanpaolo", "UCG.MI - Unicredit", "ENI.MI - Eni",
        "LDO.MI - Leonardo", "TIT.MI - Telecom Italia", "MONC.MI - Moncler",
        "RNO.PA - Renault", "MC.PA - LVMH", "AIR.PA - Airbus",
        "BTC-USD - Bitcoin", "ETH-USD - Ethereum", "SOL-USD - Solana"
    ]
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500_list = (sp500['Symbol'] + " - " + sp500['Security']).tolist()
        final_list = custom_list + sp500_list
    except:
        final_list = custom_list + ["AAPL - Apple", "MSFT - Microsoft", "TSLA - Tesla", "NVDA - Nvidia", "AMZN - Amazon", "GOOGL - Google"]
    return sorted(final_list)

ticker_options = get_ticker_list()

# --- 4. INTERFACCIA ---
st.markdown('<p class="big-title">STX ANALYZER</p>', unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])
with col1:
    selected_item = st.selectbox(
        "Seleziona Stock", options=ticker_options, index=None, 
        placeholder="Cerca (es. Tesla, Ferrari...)", label_visibility="collapsed"
    )
with col2:
    search_pressed = st.button("➔")

target_ticker = None
if selected_item:
    target_ticker = selected_item.split(" - ")[0]

start_analysis = False
if target_ticker and (search_pressed or selected_item):
    start_analysis = True

# --- 5. MOTORE AI ---
def add_fundamental_proxies(df):
    df['Volatility'] = df['Close'].rolling(window=10).std()
    vol_ma = df['Volume'].rolling(window=50).mean()
    df['News_Impact'] = df['Volume'] / vol_ma
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd.iloc[:, 0]
    df.dropna(inplace=True)
    return df

if start_analysis:
    progress_text = f"Analisi AI su {target_ticker}..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        my_bar.progress(10, text="Download dati finanziari...")
        data = yf.download(target_ticker, period="5y", interval="1d", progress=False)
        
        if len(data) < 200:
            st.error(f"Dati insufficienti per {target_ticker}.")
            my_bar.empty()
        else:
            my_bar.progress(25, text="Calcolo indicatori tecnici...")
            df = data.copy()
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = add_fundamental_proxies(df)
            
            feature_columns = ['Close', 'Volatility', 'News_Impact', 'RSI', 'MACD']
            dataset = df[feature_columns].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            prediction_days = 90 
            x_train, y_train = [], []
            for i in range(prediction_days, len(scaled_data)):
                x_train.append(scaled_data[i-prediction_days:i]) 
                y_train.append(scaled_data[i, 0]) 

            x_train, y_train = np.array(x_train), np.array(y_train)

            my_bar.progress(40, text="Attivazione Rete Neurale...")
            model = Sequential()
            model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
            model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(units=50, return_sequences=False)))
            model.add(Dropout(0.3))
            model.add(Dense(units=25, activation='relu'))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='huber')

            my_bar.progress(50, text="Addestramento Modello (Attendi)...")
            callbacks = [EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)]
            model.fit(x_train, y_train, epochs=8, batch_size=32, callbacks=callbacks, verbose=0)
            
            my_bar.progress(80, text="Calcolo proiezioni future...")
            future_days = 30
            last_sequence = scaled_data[-prediction_days:]
            current_batch = last_sequence.reshape((1, prediction_days, len(feature_columns)))
            
            future_predictions = []
            recent_volatility = np.std(df['Close'].iloc[-30:]) / df['Close'].iloc[-1]

            for i in range(future_days):
                pred_price_scaled = model.predict(current_batch, verbose=0)[0, 0]
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
            future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, future_days + 1)]

            my_bar.progress(100, text="Fatto!")
            my_bar.empty()

            # --- GRAFICO (ADAPTIVE COLORS) ---
            fig = go.Figure()
            # Storico in GRIGIO così si vede su bianco E nero
            fig.add_trace(go.Scatter(x=df.index[-180:], y=df['Close'][-180:], mode='lines', name='Storico', line=dict(color='#888888', width=2)))
            # Previsione in VERDE NEON
            fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Previsione AI', line=dict(color='#00ff41', width=3, dash='dot')))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', # Sfondo Trasparente
                plot_bgcolor='rgba(0,0,0,0)',  # Sfondo Grafico Trasparente
                font=dict(family="JetBrains Mono"),
                xaxis=dict(showgrid=False), 
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'), # Griglia sottile
                showlegend=False, margin=dict(l=20,r=20,t=20,b=20), height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            curr_price = df['Close'].iloc[-1]
            target_price = future_prices[-1]
            var_pct = ((target_price - curr_price) / curr_price) * 100
            
            # Testo statistiche con colore automatico (non specifichiamo nero/bianco)
            st.markdown(f"""
            <div style="text-align: center; margin-top: -10px; font-size: 0.9rem; opacity: 0.8;">
                ANALYSIS: {target_ticker}<br>
                Current: <b>{curr_price:.2f}€</b> &nbsp;|&nbsp; 
                Target (30d): <b>{target_price:.2f}€</b> &nbsp;|&nbsp; 
                Trend: <b style="color: {'#00ff41' if var_pct > 0 else '#ff4b4b'}">{var_pct:+.2f}%</b>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Errore tecnico: {e}")
