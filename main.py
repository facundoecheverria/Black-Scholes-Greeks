import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from scipy.integrate import quad


def black_scholes(S, K, T, r, sigma, option_type):
    """
    :param S: Spot price
    :param K: Strike price
    :param T: Time to maturity
    :param r: Risk-free rate
    :param sigma: Volatility
    :param option_type: 'call' or 'put'
    :return: Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type[0].lower() == 'c':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) * np.exp(-r*T)


def implied_volatility(S, K, T, r, Price, option_type='call', sigma=0.2, tol=1e-5, max_iter=1000):
    """
    :param S: Spot price
    :param K: Strike price
    :param T: Time to maturity
    :param r: Risk-free rate
    :param Price: Option price
    :param option_type: 'call' or 'put'
    :param sigma: Initial guess for implied volatility
    :param tol: Tolerance for convergence, indica el grado de tolerancia de diff respecto al precio.
    :param max_iter: Max number of iter
    :return: Implied volatility
    """
    i = 0
    while True:
        est = black_scholes(S, K, T, r, sigma, option_type)
        vega = black_scholes_vega(S, K, T, r, sigma)
        diff = Price - est
        if abs(diff) < tol:
            return sigma
        sigma = sigma + (diff / vega)
        if i > max_iter:
            return sigma
        i += 1


phi = lambda x: (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * x ** 2).all()  # normal distribution

integrand = lambda y: np.exp(-0.5 * (y ** 2))

psi = lambda z: 1 - (1 / (np.sqrt(2 * np.pi))) * quad(integrand, z, np.inf)[0]  # cumulative normal distribution


def bs_delta(S, K, t, sigma, r=0.01, type='call'):
    d1 = (np.log((S / K)) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    if type[0].lower() == 'c':
        return norm.cdf(d1) * np.exp(-r*t)
    else:
        return norm.cdf(-d1) * -np.exp(-r*t)


def bs_gamma(S, K, T, sigma, r=0.001, q=0.001):
    d1 = (np.log((S / K)) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * (norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_theta(S, K, T, sigma, r=0.01, q=0.01, type='call'):
    d1 = (np.log((S / K)) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))

    term_1 = -np.exp(-q * T) * ((S * norm.pdf(d1) * sigma) / 2 * np.sqrt(T))
    term_2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    term_3 = q * S * np.exp(-q * T) * norm.cdf(d1)
    if type[0].lower() == 'c':
        return term_1 - term_2 + term_3
    else:
        term_2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        term_3 = q * S * np.exp(-q * T) * norm.cdf(-d1)
        return term_1 + term_2 - term_3


def bs_rho(S, K, T, sigma, r=0.01, q=0.01, type='call'):
    d1 = (np.log((S / K)) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    if type[0].lower() == 'c':
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * (1-norm.cdf(d2))


def main():

    st.set_page_config(page_title="Black-Scholes Greeks", layout="wide")

    st.title('Black-Scholes')

    r_0, r_1, r_2 = st.columns(3)
    with r_0:
        S_ = st.number_input(label="Spot", value=100, step=1)
    with r_1:
        K_ = st.number_input('Strike price', value=100, step=1)
    with r_2:
        T = st.number_input('Time to maturity (days)', value=30, step=1)

    r_3, r_4, r_5 = st.columns(3)
    with r_3:
        r = st.number_input('Risk-free rate (%)', value=1)
    with r_4:
        sigma = st.number_input('Volatility (%)', value=20, step=1)
    with r_5:
        option_type = st.selectbox('Option type', ['Call', 'Put'], index=0)

    with st.expander("Plot settings"):
        r_6, r_7, r_8 = st.columns(3)

        with r_6:
            _range = st.number_input("+/- % Strike Range", value=30) / 100
        with r_7:
            points = st.number_input("∆x (mayor -> mas lento)", value=0.5)
        with r_8:
            x_axis = st.selectbox("X-Axis", ["Spot", "Strike"], index=0)

    option_type = option_type.lower()


    if x_axis == "Strike":
        ks = np.arange(K_ - (K_ * _range), K_ + (K_ * _range), points)
        xs = ks
        S = np.array([float(S_) for _ in range(len(ks))])
        pnl_axis_title = "Price"
    else:
        S = np.arange(S_ - (S_ * _range), S_ + (S_ * _range), points)
        ks = np.array([float(K_) for _ in range(len(S))])
        xs = S
        pnl_axis_title = "PnL"

    df = pd.DataFrame(
        {"x-axis": xs,
         "Strike": ks,
         "Price": black_scholes(S, K=ks, T=(T / 252), r=(r / 100), sigma=(sigma / 100), option_type=option_type),
         "Delta": bs_delta(S, K=ks, t=(T / 252), sigma=(sigma / 100), r=(r / 100), type=option_type),
         "Gamma": bs_gamma(S, K=ks, T=(T / 252), sigma=(sigma / 100), r=(r / 100), q=(r / 100)),
         "Theta": bs_theta(S, K=ks, T=(T / 252), sigma=(sigma / 100), r=(r / 100), q=(r / 100), type=option_type),
         "Vega": black_scholes_vega(S, K=ks, T=(T / 252), r=(r / 100), sigma=(sigma / 100)),
         "Rho": bs_rho(S, K=ks, T=(T / 252), sigma=(sigma / 100), r=(r / 100), q=(r / 100), type=option_type)})

    df['Theta'].clip(upper=0, inplace=True)
    plt.style.use('ggplot')

    fig, ((ax1, ax2),
          (ax3, ax4),
          (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(22, 16), dpi=200, sharex=False, sharey=False)
    ax1.plot(df['x-axis'], df['Price'], lw=3)
    ax1.axvline(S_, linestyle="--", c="gray", label="Spot")
    ax1.axvline(K_, linestyle="--", c="Blue", label="K")
    ax1.set_xlabel(x_axis)
    ax1.set_title(pnl_axis_title)
    ax1.legend()

    ax2.plot(df['x-axis'], df['Delta'], lw=3)
    ax2.axvline(S_, linestyle="--", c="gray", label="Spot")
    ax2.axvline(K_, linestyle="--", c="Blue", label="K")
    ax2.set_xlabel(x_axis)
    ax2.set_title("Delta")
    ax2.legend()

    ax3.plot(df['x-axis'], df['Gamma'], lw=3)
    ax3.axvline(S_, linestyle="--", c="gray", label="Spot")
    ax3.axvline(K_, linestyle="--", c="Blue", label="K")
    ax3.set_xlabel(x_axis)
    ax3.set_title("Gamma")
    ax3.legend()

    ax4.plot(df['x-axis'], df['Vega'], lw=3)
    ax4.axvline(S_, linestyle="--", c="gray", label="Spot")
    ax4.axvline(K_, linestyle="--", c="Blue", label="K")
    ax4.set_xlabel(x_axis)
    ax4.set_title("Vega")
    ax4.legend()

    ax5.plot(df['x-axis'], df['Theta'], lw=3)
    ax5.axvline(S_, linestyle="--", c="gray", label="Spot")
    ax5.axvline(K_, linestyle="--", c="Blue", label="K")
    ax5.set_xlabel(x_axis)
    ax5.set_title("Theta")
    ax5.legend()

    ax6.plot(df['x-axis'], df['Rho'], lw=3)
    ax6.axvline(S_, linestyle="--", c="gray", label="Spot")
    ax6.axvline(K_, linestyle="--", c="Blue", label="K")
    ax6.set_xlabel(x_axis)
    ax6.set_title("Rho")
    ax6.legend()

    st.pyplot(fig)

    df[str(x_axis)] = df['x-axis']
    df.drop(columns=["x-axis"], inplace=True)

    st.download_button(label="Descargar datos", data=df.to_csv(index=False), file_name="Black-Scholes.csv",
                       mime="text/csv")

    st.dataframe(df)


if __name__ == '__main__':
    main()
