import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

# Page setup
st.set_page_config(
    page_title="Premium Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
}
.metric-card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.08);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
    text-align: center;
}
.big-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #00E5FF;
}
.subtitle {
    text-align: center;
    color: #dcdcdc;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="big-title">📊 Smart Stock Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict future stock prices with AI-powered ARIMA forecasting</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙ Control Panel")
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
years = st.sidebar.slider("Select Historical Data", 1, 10, 5)
forecast_days = st.sidebar.slider("Forecast Period", 30, 365, 180)

if st.sidebar.button("🚀 Generate Forecast"):

    with st.spinner("Loading market data..."):
        df = yf.download(
            ticker,
            period=f"{years}y",
            interval="1d",
            auto_adjust=True
        )

    if df.empty:
        st.error("No stock data found.")
    else:
        close = df["Close"]

        # Moving averages
        df["MA50"] = close.rolling(50).mean()
        df["MA200"] = close.rolling(200).mean()

        # Train ARIMA
        model = ARIMA(close, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_days)

        future_dates = pd.date_range(
            start=close.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days
        )

        forecast_df = pd.DataFrame({
            "Forecast": forecast.values
        }, index=future_dates)

        latest_price = round(close.iloc[-1], 2)
        predicted_price = round(forecast.iloc[-1], 2)
        growth = round(((predicted_price - latest_price) / latest_price) * 100, 2)

        # KPI Section
        col1, col2, col3 = st.columns(3)

        col1.markdown(f"""
        <div class="metric-card">
        <h3>💰 Current Price</h3>
        <h1>${latest_price}</h1>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-card">
        <h3>🎯 Predicted Price</h3>
        <h1>${predicted_price}</h1>
        </div>
        """, unsafe_allow_html=True)

        trend = "📈 BUY" if growth > 0 else "📉 SELL"

        col3.markdown(f"""
        <div class="metric-card">
        <h3>Trend Signal</h3>
        <h1>{trend}</h1>
        <p>{growth}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Historical Chart
        st.subheader("📊 Historical Stock Performance")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=close.index,
            y=close,
            mode='lines',
            name='Close Price'
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["MA50"],
            mode='lines',
            name='50-Day MA'
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["MA200"],
            mode='lines',
            name='200-Day MA'
        ))

        fig.update_layout(
            template="plotly_dark",
            height=600,
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Forecast Chart
        st.subheader("🔮 Future Forecast")

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=close[-180:].index,
            y=close[-180:],
            mode='lines',
            name='Historical'
        ))

        fig2.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Forecast"],
            mode='lines',
            name='Forecast'
        ))

        fig2.update_layout(
            template="plotly_dark",
            height=600,
            hovermode="x unified"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Forecast Table
        st.subheader("📅 Forecast Data")
        st.dataframe(
            forecast_df.style.highlight_max(axis=0)
        )

        # Download
        csv = forecast_df.to_csv().encode('utf-8')

        st.download_button(
            "⬇ Download Forecast CSV",
            csv,
            f"{ticker}_forecast.csv",
            "text/csv"
        )

        # Market sentiment bar
        st.subheader("📢 Market Sentiment")
        sentiment_score = min(max((growth + 50) / 100, 0), 1)
        st.progress(sentiment_score)

        if growth > 10:
            st.success("Strong bullish outlook 🚀")
        elif growth > 0:
            st.info("Moderate growth expected 📈")
        else:
            st.warning("Possible decline ahead ⚠")
