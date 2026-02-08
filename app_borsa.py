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
st.set_page_config(page_title="STX Ultimate", page_icon="💎", layout="centered")

# --- 2. CSS STILE PRO ---
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
st.markdown('<p class="subtitle">Macro-Economics + Deep History + Stability Control</p>', unsafe_allow_html=True)

ticker_input = st.text_input(
    "Inserisci Ticker", 
    placeholder="Es. LDO.MI, TSLA, BTC-USD, ENI.MI", 
    help="Analisi correllata con Oro, Petrolio, Tassi e Volatilità."
).upper().strip()

# --- 4. MOTORE IBRIDO (MACRO + STABILITY) ---
PREDICTION_DAYS = 90    
FUTURE_DAYS = 365       

@st.cache_data(ttl=12*3600)
def get_ultimate_data(ticker):
    try:
        # 1. Scarica l'Azione Target (MAX History)
        stock = yf.download(ticker, period="max", interval="1d", progress=False)
        
        if len(stock) < 300: return None, None, None, None

        # 2. Scarica i Fattori Macro (Il "Cervello" esterno)
        # ^VIX: Paura | GC=F: Oro/Guerra | CL=F: Energia | ^TNX: Tassi | ^GSPC: Mercato
        tickers = ["^VIX", "GC=F", "CL=F", "^TNX", "^GSPC"]
        macro_data = yf.download(tickers, period="max", interval="1d", progress=False)['Close']
        
        # Pulizia
        if isinstance(stock.columns, pd.MultiIndex): stock.columns = stock.columns.get_level_values(0)
        
        # 3. Unione Dataset
        df = stock[['Close']].rename(columns={'Close': 'Stock_Price'})
        df = df.join(macro_data, how='inner') # Allinea le date
        
        df.rename(columns={
            '^VIX': 'Fear_Index', 'GC=F': 'Gold_War', 'CL=F': 'Oil_Energy', 
            '^TNX': 'Rates_Inflation', '^GSPC': 'General_Market'
        }, inplace=True)

        # 4. Calcolo Variazioni % (Input per l'AI)
        df_pct = df.pct_change().fillna(0)
        
        # Volatilità (per le zone rosse)
        df_pct['Stock_Vol'] = df_pct['Stock_Price'].rolling(20).std().fillna(0)

        # 5. Calcolo Forza Relativa vs S&P500 (Feature dello Script 1)
        # Lo calcoliamo "fuori" dall'AI per mostrarlo nelle statistiche
        try:
            market_cum = (1 + df_pct['General_Market']).cumprod()
            stock_cum = (1 + df_pct['Stock_Price']).cumprod()
            # Conversione float per evitare errori rossi
            val = stock_cum.iloc[-1] - market_cum.iloc[-1]
            relative_strength = float(val)
        except:
            relative_strength = 0.0

        # 6. Correlazioni (Feature dello Script 2 per il grafico a barre)
        recent_corr = df_pct.iloc[-500:].corr()['Stock_Price'].drop(['Stock_Price', 'Stock_Vol'])

        return df, df_pct, recent_corr, relative_strength

    except Exception as e:
        return None, None, None, None

@st.cache_resource(show_spinner=False)
def train_ultimate_model(df_pct):
    # Usiamo TUTTI i fattori per addestrare
    feature_cols = ['Stock_Price', 'Fear_Index', 'Gold_War', 'Oil_Energy', 'Rates_Inflation', 'General_Market']
    data_values = df_pct[feature_cols].values

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data_values)

    x_train, y_train = [], []
    for i in range(PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[i-PREDICTION_DAYS:i]) 
        y_train.append(scaled_data[i, 0]) # Target: Stock Price

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Modello LSTM Potente
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2]))) 
    model.add(LSTM(units=60, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=60))
    model.add(Dropout(0.3))
    model.add(Dense(units=30, activation='relu'))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    # Batch size ottimizzato
    model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=0)
    
    return model, scaler, scaled_data, feature_cols

# --- ESECUZIONE ---
if ticker_input:
    progress_bar = st.progress(0, text="Scansione Macroeconomica Globale...")
    
    # Recupero Dati Ultimate
    df_prices, df_pct, correlations, rel_strength = get_ultimate_data(ticker_input)
    
    if df_prices is None:
        st.error("Dati insufficienti o Ticker non valido.")
        progress_bar.empty()
    else:
        # 1. MOSTRA CORRELAZIONI (Intelligence Script 2)
        st.markdown("##### 🧠 Analisi Sensibilità: Cosa muove il prezzo?")
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
        progress_bar.progress(40, text="Training su Correlazioni Storiche...")
        model, scaler, scaled_data, feature_cols = train_ultimate_model(df_pct)
        
        # 3. PROIEZIONE CON DECAY (Stability Script 1)
        progress_bar.progress(70, text="Simulazione Scenari Futuri...")
        
        last_sequence = scaled_data[-PREDICTION_DAYS:]
        current_batch = last_sequence.reshape((1, PREDICTION_DAYS, len(feature_cols)))
        future_stock_returns = []
        
        # Trend inerziale degli altri fattori (Gold, Oil, ecc.)
        recent_macro_trend = np.mean(scaled_data[-30:, 1:], axis=0) 
        
        # FRENO A MANO (Decay Factor dallo Script 1 per evitare razzi)
        decay = 0.998 

        for i in range(FUTURE_DAYS):
            pred_stock_ret = model.predict(current_batch, verbose=0)[0, 0]
            
            # Applichiamo il freno
            pred_stock_ret *= (decay ** i)
            
            future_stock_returns.append(pred_stock_ret)
            
            # Aggiorniamo il batch con la previsione stock + trend macro
            new_macro_values = recent_macro_trend + np.random.normal(0, 0.01, size=len(recent_macro_trend))
            new_row = np.insert(new_macro_values, 0, pred_stock_ret)
            current_batch = np.append(current_batch[:, 1:, :], [[new_row]], axis=1)

        # Ricostruzione Prezzi
        dummy_matrix = np.zeros((len(future_stock_returns), len(feature_cols)))
        dummy_matrix[:, 0] = future_stock_returns
        future_pcts = scaler.inverse_transform(dummy_matrix)[:, 0]

        last_price = df_prices['Stock_Price'].iloc[-1]
        future_prices = []
        curr_p = last_price
        
        for ret in future_pcts:
            curr_p = curr_p * (1 + ret)
            future_prices.append(curr_p)
            
        last_date = df_prices.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, FUTURE_DAYS + 1)]

        progress_bar.progress(100, text="Analisi Completa.")
        progress_bar.empty()

        # 4. GRAFICO FINALE
        fig = go.Figure()
        
        # Storico
        past = df_prices.iloc[-365:]
        fig.add_trace(go.Scatter(x=past.index, y=past['Stock_Price'], mode='lines', name='Storico', line=dict(color='var(--text-color)', width=2)))
        
        # Zone Volatilità (Script 1 Feature)
        high_vol = df_pct['Stock_Vol'].iloc[-365:].quantile(0.95)
        fig.add_trace(go.Scatter(
            x=past.index,
            y=[past['Stock_Price'].max() if v > high_vol else None for v in df_pct['Stock_Vol'].iloc[-365:]],
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
        
        # 5. DATI FINALI (Trend + Forza Relativa Script 1)
        chg = ((future_prices[-1] - last_price) / last_price) * 100
        
        if rel_strength > 0: rs_color, rs_sign = "#00ff00", "+"
        else: rs_color, rs_sign = "#ff4444", ""
        
        st.markdown(f"""
        <div style="text-align:center; font-size:1.1rem;">
            Target 1Y: <b>{future_prices[-1]:.2f}</b> | Trend: <b style='color:{'#00ff00' if chg>0 else '#ff4444'}'>{chg:+.2f}%</b> <br>
            <span style="font-size:0.9rem; color:gray">Forza Relativa vs S&P500: <b style="color:{rs_color}">{rs_sign}{rel_strength*100:.2f}%</b></span>
        </div>
        """, unsafe_allow_html=True)
