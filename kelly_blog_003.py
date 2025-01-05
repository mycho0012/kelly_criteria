import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

def simulate_final_value(returns, f):
    """
    일별 수익률(returns)에 대해, 매일 자본의 'f' 비중을 베팅한다고 가정하고
    최종 자본을 구한다. 파산(자본<=0) 시 즉시 0으로 반환.
    """
    capital = 1.0
    
    # Ensure array of float64
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy(dtype=np.float64)
    elif isinstance(returns, (list, tuple)):
        returns = np.array(returns, dtype=np.float64)
    
    try:
        for r in returns:
            capital *= (1 + f * r)
            if capital <= 0:
                return 0.0
        return capital
    except Exception as e:
        st.error(f"수익률 계산 중 오류 발생: {str(e)}")
        return 0.0

def annualized_growth_ratio(final_capital, days):
    """
    최종 자본(final_capital)에서 연평균 성장률(기하평균)을 추정.
    (final_capital^(252/days)) - 1
    """
    if final_capital <= 0:
        return -1.0
    try:
        annual_factor = 252 / max(days, 1)
        growth = (final_capital ** annual_factor) - 1
        return growth
    except Exception as e:
        st.error(f"연간 성장률 계산 중 오류 발생: {str(e)}")
        return -1.0

def kelly_sweep(returns, f_min=-0.1, f_max=0.5, steps=61):
    """
    f_min ~ f_max 범위를 steps만큼 등분하여,
    각 f값에 대해 최종자본 -> 연평균 성장률(기하평균)을 계산.
    """
    try:
        # Convert to float64 array
        if isinstance(returns, pd.Series):
            returns = returns.to_numpy(dtype=np.float64)
        elif isinstance(returns, (list, tuple)):
            returns = np.array(returns, dtype=np.float64)
        
        # Remove infinities or NaN
        returns = returns[np.isfinite(returns)]
        
        f_values = np.linspace(f_min, f_max, steps)
        n_days = len(returns)
        
        results = []
        for f in f_values:
            final_cap = simulate_final_value(returns, float(f))
            growth = annualized_growth_ratio(final_cap, n_days)
            results.append((float(f), growth))
        
        return results
    
    except Exception as e:
        st.error(f"Kelly sweep 계산 중 오류 발생: {str(e)}")
        # Return fallback
        return [(0.0, -1.0)]

def main():
    st.title("Kelly Criterion Sweep Simulation")

    st.sidebar.header("Simulation Settings")
    ticker = st.sidebar.text_input("종목 티커(예: 005930.KS (삼성전자))", value="005930.KS")
    start_date_val = st.sidebar.date_input("시작 날짜", value=date(2018,1,1))
    end_date_val = st.sidebar.date_input("끝 날짜", value=date.today())
    
    st.sidebar.write("---")
    f_min = st.sidebar.slider("f_min (최소 베팅 비율)", 
                              min_value=-0.5, max_value=0.0, 
                              value=-0.1, step=0.01)
    f_max = st.sidebar.slider("f_max (최대 베팅 비율)", 
                              min_value=0.0, max_value=2.0, 
                              value=0.5, step=0.01)
    steps = st.sidebar.slider("단계 수 (steps)", 
                              min_value=10, max_value=200, 
                              value=61, step=1)
    
    # Ticker Info
    try:
        ticker_info = yf.Ticker(ticker)
        company_name = ticker_info.info.get('longName', 'Unknown Company')
    except Exception as e:
        st.error(f"Ticker 정보를 가져오는 중 오류 발생: {e}")
        company_name = "Unknown Company"
    
    st.write(f"**티커**: {ticker} ({company_name}), "
             f"**기간**: {start_date_val} ~ {end_date_val}")

    # 1) Download Data
    try:
        df = yf.download(
            ticker, 
            start=str(start_date_val), 
            end=str(end_date_val), 
            progress=False
        )
        if df.empty:
            st.error("해당 기간에 유효한 데이터가 없습니다. 날짜/티커를 다시 확인해주세요.")
            return
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류 발생: {e}")
        return
    
    # Drop NaN
    df = df.dropna(how='any')
    if df.empty:
        st.error("유효한 행이 없는 데이터프레임입니다.")
        return
    
    # 2) Price Selection
    if 'Adj Close' in df.columns:
        price_data = df['Adj Close']
    elif 'Close' in df.columns:
        price_data = df['Close']
    else:
        st.error("No 'Adj Close' or 'Close' column found in data.")
        return
    
    # Make sure it's a Series, remove multi-dim
    price_data = price_data.squeeze()

    # Ensure index is datetime and sorted
    price_data.index = pd.to_datetime(price_data.index, errors='coerce')
    price_data.sort_index(inplace=True)
    
    # Remove time zone if present
    if price_data.index.tz is not None:
        price_data.index = price_data.index.tz_localize(None)

    # Convert to numeric
    price_data = pd.to_numeric(price_data, errors='coerce').dropna()
    if price_data.empty:
        st.error("유효한 숫자형 가격 데이터가 없습니다.")
        return

    st.write(f"가져온 데이터 개수: {len(price_data)}")
    
    # Quick debugging info (optional)
    st.write("First 5 Rows:")
    st.write(price_data.head())
    st.write("Index dtype:", price_data.index.dtype)
    
    # Plot with line_chart
    if len(price_data) < 2:
        st.warning("Chart cannot be displayed because there are fewer than 2 data points.")
    else:
        st.line_chart(price_data, height=300, use_container_width=True)

    # 3) Returns
    returns = price_data.pct_change().dropna()
    if len(returns) < 2:
        st.warning("수익률 계산 가능한 데이터가 부족합니다.")
        return

    # 4) Kelly Sweep
    sweep_result = kelly_sweep(returns, f_min=f_min, f_max=f_max, steps=steps)
    if not sweep_result:
        st.error("켈리 스윕 결과가 비어있습니다.")
        return
    
    f_vals = [r[0] for r in sweep_result]
    growth_vals = [r[1] for r in sweep_result]

    idx_max = np.argmax(growth_vals)
    best_f = f_vals[idx_max]
    best_growth = growth_vals[idx_max] * 100.0

    st.subheader("결과 요약")
    if best_growth < 0:
        st.write("해당 구간에서 최대 성장률이 음수입니다. (시장 하락 or 파산 위험이 큼)")
        st.write(f"최대 성장률: {best_growth:.2f}%, f = {best_f:.2%}")
    else:
        st.write(f"**최적 Kelly 비율**: {best_f:.2%}, "
                 f"**연평균성장률**: {best_growth:.2f}%")

    # 5) Plot Sweep with Matplotlib
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot([fv*100 for fv in f_vals], [gv*100 for gv in growth_vals],
            label='Annual Growth Rate')
    
    ax.axvline(best_f*100, color='red', linestyle='--',
               label=f'Best f={best_f*100:.1f}%')
    ax.scatter([best_f*100], [best_growth], color='red')
    
    ax.set_xlabel("Wagered fraction f (%)")
    ax.set_ylabel("Annualized Growth (%)")
    ax.set_title(f"Kelly Sweep for {ticker} ({company_name})")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

if __name__ == "__main__":
    main()
