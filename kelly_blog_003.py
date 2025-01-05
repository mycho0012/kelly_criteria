import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

def simulate_final_value(returns, f):
    """
    일별 수익률(returns)에 대해, 매일 자본의 'f' 비중을 베팅한다고 가정하고
    최종 자본(=계좌가치)을 구한다.
    초기 자본=1, daily_update: capital_{t+1} = capital_t * (1 + f*return_t)
    파산(자본<=0) 시 즉시 0으로 반환.
    """
    capital = 1.0
    # Convert returns to numpy array if it's a pandas Series
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    for r in returns:
        capital *= (1 + f*r)
        if capital <= 0:
            return 0.0
    return capital

def annualized_growth_ratio(final_capital, days):
    """
    final_capital(최종 자본)에서 연평균 성장률(기하평균)을 추정.
    (final_capital^(252/days)) - 1
    """
    if final_capital <= 0:
        return -1.0  # 파산이면 -100%로 보정
    annual_factor = 252 / days
    growth = (final_capital ** annual_factor) - 1
    return growth

def kelly_sweep(returns, f_min=-0.1, f_max=0.5, steps=61):
    """
    f_min ~ f_max 범위를 steps만큼 등분해,
    각 f값에 대해 최종자본 -> 연평균 성장률을 계산.
    """
    # Convert returns to numpy array if it's a pandas Series
    if isinstance(returns, pd.Series):
        returns = returns.values
        
    f_values = np.linspace(f_min, f_max, steps)
    n_days = len(returns)
    
    results = []
    for f in f_values:
        final_cap = simulate_final_value(returns, f)
        growth = annualized_growth_ratio(final_cap, n_days)  # 기하평균
        results.append((f, growth))
    
    return results

def main():
    st.title("Kelly Criterion Sweep Simulation")

    # 사용자 입력 파라미터
    st.sidebar.header("Simulation Settings")
    ticker = st.sidebar.text_input("종목 티커(예: 005930.KS (삼성전자))", value="005930.KS")
    start_date = st.sidebar.date_input("시작 날짜", value=date(2018,1,1))
    end_date = st.sidebar.date_input("끝 날짜", value=date.today())
    
    st.sidebar.write("---")
    f_min = st.sidebar.slider("f_min (최소 베팅 비율)", min_value=-0.5, max_value=0.0, value=-0.1, step=0.01)
    f_max = st.sidebar.slider("f_max (최대 베팅 비율)", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    steps = st.sidebar.slider("단계 수 (steps)", min_value=10, max_value=200, value=61, step=1)
    
    # Ticker 이름 가져오기
    try:
        ticker_info = yf.Ticker(ticker)
        company_name = ticker_info.info.get('longName', 'Unknown Company')
    except Exception as e:
        st.error(f"Ticker 정보를 가져오는 중 오류 발생: {e}")
        company_name = "Unknown Company"
    
    st.write(f"**티커**: {ticker} ({company_name}), **기간**: {start_date} ~ {end_date}")

    # 1) 데이터 불러오기
    try:
        df = yf.download(ticker, start=str(start_date), end=str(end_date), progress=False)
        if df.empty:
            st.error("해당 기간에 유효한 데이터가 없습니다. 날짜/티커를 다시 확인해주세요.")
            return
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류 발생: {e}")
        return
    
    # 조정종가만 사용
    df = df.dropna()
    if 'Adj Close' in df.columns:
        price = df['Adj Close']
    else:
        # yfinance 반환 형태가 다를 경우 대비
        price = df['Close']

    st.write(f"가져온 데이터 개수: {len(price)}")
    st.line_chart(price, height=200, use_container_width=True)

    # 2) 수익률 계산
    returns = price.pct_change().dropna()
    if len(returns) < 2:
        st.warning("수익률 계산 가능한 데이터가 부족합니다.")
        return

    # 3) 켈리 스윕
    sweep_result = kelly_sweep(returns, f_min=f_min, f_max=f_max, steps=steps)
    f_vals = [r[0] for r in sweep_result]
    growth_vals = [r[1] for r in sweep_result]

    # 최적 f
    idx_max = np.argmax(growth_vals)
    best_f = f_vals[idx_max]
    best_growth = growth_vals[idx_max] * 100.0  # %

    st.subheader("결과 요약")
    if best_growth < 0:
        st.write("해당 구간에서 최대 성장률이 음수입니다. (시장 하락 or 파산 위험이 큼)")
        st.write(f"최대 성장률: {best_growth:.2f}%, f = {best_f:.2%}")
    else:
        st.write(f"**최적 Kelly 비율**: {best_f:.2%}, **연평균성장률**: {best_growth:.2f}%")

    # 그래프 그리기 (matplotlib -> st.pyplot)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot([f*100 for f in f_vals], [g*100 for g in growth_vals], label='Annual Growth Rate')
    ax.axvline(best_f*100, color='red', linestyle='--', label=f'Best f={best_f*100:.1f}%')
    ax.scatter([best_f*100], [best_growth], color='red')
    
    ax.set_xlabel("Wagered fraction f (%)")
    ax.set_ylabel("Annualized Growth (%)")
    ax.set_title(f"Kelly Sweep for {ticker} ({company_name})")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

if __name__ == "__main__":
    main()
