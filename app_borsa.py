import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="STX Analyzer", page_icon="📈", layout="centered")

# --- CSS MINIMAL (JETBRAINS MONO + OMBRE) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace !important; }
        
        /* Titolo */
        .big-title {
            text-align: center; font-size: 3rem !important; font-weight: 700;
            margin-bottom: 10px; letter-spacing: -1px;
        }
        
        /* Barra di ricerca */
        .stTextInput > div > div > input {
            text-align: center; font-size: 1.2rem; border-radius: 12px;
            padding: 15px; border: 1px solid #e0e0e0;
            box-shadow: 0px 10px 20px rgba(0,0,0,0.05); transition: all 0.3s ease;
        }
        .stTextInput > div > div > input:focus { border-color: #000; box-shadow: 0px 15px 25px rgba(0,0,0,0.1); }
        
        /* Grafico */
        .stPlotlyChart {
            background-color: white; border-radius: 15px; padding: 10px;
            box-shadow: 0px 15px 40px rgba(0,0,0,0.15); margin-top: 30px;
        }
        
        /* Nascondi elementi inutili */
        #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- UI ---
st.markdown('<p class="big-title">STX ANALYZER</p>', unsafe_allow_html=True)
ticker_input = st.text_input("", placeholder="Insert stock ticker (es. STLA, RACE.MI)", help="Premi Invio")

# --- LOGICA ---
if ticker_input:
    with st.spinner(f"Analisi AI su {ticker_input}..."):
        try:
            # Scarico dati (2 anni per velocità cloud)
            data = yf.download(ticker_input, period="2y", interval="1d", progress=False)
            
            if len(data) < 60:
                st.error("Dati insufficienti o simbolo errato.")
            else:
                df = data.copy()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                # Scaler
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df[['Close']].values)
                
                # Preparazione Training
                prediction_days = 60
                x_train, y_train = [], []
                for i in range(prediction_days, len(scaled_data)):
                    x_train.append(scaled_data[i-prediction_days:i, 0])
                    y_train.append(scaled_data[i, 0])
                
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                
                # Modello AI Leggero (per il Cloud gratuito)
                model = Sequential()
                model.add(Input(shape=(x_train.shape[1], 1)))
                model.add(LSTM(units=50, return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                
                # Training veloce (3 epoche)
                model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
                
                # Previsione Futura (30 giorni)
                future_days = 30
                test_input = scaled_data[-prediction_days:].reshape(1, prediction_days, 1)
                future_preds = []
                
                for _ in range(future_days):
                    pred = model.predict(test_input, verbose=0)[0][0]
                    future_preds.append(pred)
                    test_input = np.append(test_input[:, 1:, :], [[[pred]]], axis=1)
                
                future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
                
                # Date
                last_date = df.index[-1]
                future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
                
                # --- GRAFICO FINALE ---
                fig = go.Figure()
                
                # Passato (Nero)
                fig.add_trace(go.Scatter(
                    x=df.index[-120:], y=df['Close'][-120:], 
                    mode='lines', name='Past', 
                    line=dict(color='black', width=2)
                ))
                
                # Futuro (Verde Matrix)
                fig.add_trace(go.Scatter(
                    x=future_dates, y=future_preds.flatten(), 
                    mode='lines', name='Forecast', 
                    line=dict(color='#00ff41', width=3, dash='dot')
                ))
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="JetBrains Mono"),
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#eee'),
                    showlegend=False, margin=dict(l=20,r=20,t=20,b=20), height=450
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Dati finali
                start_p = df['Close'].iloc[-1]
                end_p = future_preds[-1][0]
                diff = ((end_p - start_p) / start_p) * 100
                
                st.markdown(f"""
                <div style='text-align:center; color:gray; margin-top:-10px;'>
                    Current: <b>{start_p:.2f}€</b> &nbsp;|&nbsp; 
                    Target 30d: <b>{end_p:.2f}€</b> &nbsp;|&nbsp; 
                    Trend: <b style='color:{'green' if diff>0 else 'red'}'>{diff:+.2f}%</b>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Errore: {e}. Prova con un altro ticker.")