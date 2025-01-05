import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

def simulate_final_value(returns, f):
    capital = 1.0
    # Ensure array of float64
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy(dtype=np.float64)
    elif isinstance(returns, (list, tuple)):
        returns = np.array(returns, dtype=np.float64)
    
    try:
        for r in returns:
            capital *= (1 + f*r)
            if capital <= 0:
                return 0.0
        return capital
    except Exception as e:
        st.error(f"수익률 계산 중 오류 발생: {str(e)}")
        return 0.0

def annualized_growth_ratio(final_capital, days):
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
    try:
        if isinstance(returns, pd.Series):
            returns = returns.to_numpy(dtype=np.float64)
        elif isinstance(returns, (list, tuple)):
            returns = np.array(returns, dtype=np.float64)
        
        # OPTIONAL: remove infinite or NaN
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
        # Return a fallback so the code won't crash outside
        return [(0.0, -1.0)]

def main():
    st.title("Kelly Criterion Sweep Simulation")

    st.sidebar.header("Simulation Settings")
    ticker = st.sidebar.text_input("종목 티커(예: 005930.KS)", value="005930.KS")
    start_date_input = st.sidebar.date_input("시작 날짜", value=date(2018,1,1))
    end_date_input = st.sidebar.date_input("끝 날짜", value=date.today())
    
    st.sidebar.write("---")
    f_min = st.sidebar.slider("f_min (최소 베팅 비율)", 
                              min_value=-0.5, max_value=0.0, value=-0.1, step=0.01)
    f_max = st.sidebar.slider("f_max (최대 베팅 비율)", 
                              min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    steps = st.sidebar.slider("단계 수 (steps)", min_value=10, max_value=200, value=61, step=1)
    
    # Ticker Info
    try:
        ticker_info = yf.Ticker(ticker)
        company_name = ticker_info.info.get('longName', 'Unknown Company')
    except Exception as e:
        st.error(f"Ticker 정보를 가져오는 중 오류 발생: {e}")
        company_name = "Unknown Company"
    
    st.write(f"**티커**: {ticker} ({company_name}), "
             f"**기간**: {start_date_input} ~ {end_date_input}")

    # 1) 데이터 불러오기
    try:
        df = yf.download(
            ticker, 
            start=str(start_date_input), 
            end=str(end_date_input), 
            progress=False
        )
        if df.empty:
            st.error("해당 기간에 유효한 데이터가 없습니다. 날짜/티커를 다시 확인해주세요.")
            return
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류 발생: {e}")
        return
    
    # Drop any rows with NaN
    df = df.dropna(how='any')
    if df.empty:
        st.error("유효한 행이 없는 데이터프레임입니다.")
        return
    
    # 2) 적절한 가격 열 추출
    if 'Adj Close' in df.columns:
        price_data = df['Adj Close']
    elif 'Close' in df.columns:
        price_data = df['Close']
    else:
        st.error("No 'Adj Close' or 'Close' column found in data.")
        return

    # Some yfinance data might return multi-dimensional columns (unlikely here, but possible)
    # Use .squeeze() to ensure we get a Series if it was (rows,1) shape
    price_data = price_data.squeeze()

    # Double-check dimensionality
    if price_data.ndim != 1:
        st.error(f"가격 데이터가 1차원이 아닙니다 (shape={price_data.shape}).")
        return

    # Convert to numeric and drop NaN
    price_data = pd.to_numeric(price_data, errors='coerce').dropna()
    if price_data.empty:
        st.error("유효한 숫자형 가격 데이터가 없습니다.")
        return
    
    st.write(f"가져온 데이터 개수: {len(price_data)}")
    st.line_chart(price_data, height=200, use_container_width=True)

    # 3) 수익률 계산
    returns = price_data.pct_change().dropna()
    if len(returns) < 2:
        st.warning("수익률 계산 가능한 데이터가 부족합니다.")
        return

    # 4) 켈리 스윕
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

    # 5) 성장률 곡선 시각화
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
