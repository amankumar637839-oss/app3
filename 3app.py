import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}
.metric-card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.08);
    text-align: center;
}
.big-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: cyan;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="big-title">📊 Smart Stock Forecast Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙ Control Panel")
ticker = st.sidebar.text_input("Stock Symbol", "AAPL")
years = st.sidebar.slider("Historical Data (Years)", 1, 10, 5)
forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 180)

if st.sidebar.button("🚀 Generate Forecast"):

    try:
        # Download stock data
        with st.spinner("Fetching stock data..."):
            df = yf.download(
                ticker,
                period=f"{years}y",
                interval="1d",
                auto_adjust=True
            )

        if df.empty:
            st.error("No stock data found. Check ticker symbol.")
            st.stop()

        # Fix Close column issue
        close = df["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna()

        # Moving averages
        df["MA50"] = close.rolling(window=50).mean()
        df["MA200"] = close.rolling(window=200).mean()

        # Train ARIMA
        with st.spinner("Training model..."):
            model = ARIMA(close, order=(5, 1, 0))
            model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=forecast_days)

        future_dates = pd.date_range(
            start=close.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        forecast_df = pd.DataFrame(
            {"Forecast": forecast},
            index=future_dates
        )

        # Metrics
        latest_price = round(float(close.iloc[-1]), 2)
        predicted_price = round(float(forecast.iloc[-1]), 2)
        growth = round(((predicted_price - latest_price) / latest_price) * 100, 2)

        # KPI cards
        col1, col2, col3 = st.columns(3)

        col1.markdown(f"""
        <div class="metric-card">
        <h3>💰 Current Price</h3>
        <h2>${latest_price}</h2>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-card">
        <h3>🎯 Predicted Price</h3>
        <h2>${predicted_price}</h2>
        </div>
        """, unsafe_allow_html=True)

        trend = "📈 BUY" if growth > 0 else "📉 SELL"

        col3.markdown(f"""
        <div class="metric-card">
        <h3>Trend Signal</h3>
        <h2>{trend}</h2>
        <p>{growth}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Historical chart
        st.subheader("📊 Historical Stock Performance")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Close Price"
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["MA50"],
            mode="lines",
            name="MA50"
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["MA200"],
            mode="lines",
            name="MA200"
        ))

        fig.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Forecast chart
        st.subheader("🔮 Forecast Chart")

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=close[-180:].index,
            y=close[-180:].values,
            mode="lines",
            name="Historical"
        ))

        fig2.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Forecast"].values,
            mode="lines",
            name="Forecast"
        ))

        fig2.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Forecast table
        st.subheader("📅 Forecast Data")
        st.dataframe(forecast_df)

        # CSV Download
        csv = forecast_df.to_csv().encode("utf-8")

        st.download_button(
            label="⬇ Download Forecast CSV",
            data=csv,
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )

        # Sentiment
        st.subheader("📢 Market Sentiment")

        sentiment_score = min(max((growth + 50) / 100, 0), 1)
        st.progress(sentiment_score)

        if growth > 10:
            st.success("Strong bullish outlook 🚀")
        elif growth > 0:
            st.info("Moderate growth expected 📈")
        else:
            st.warning("Possible decline ahead ⚠")

    except Exception as e:
        st.error(f"Error: {str(e)}")
