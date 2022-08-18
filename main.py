import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.abc import x
from StreamlitModels import ALL_MODELS

FIGURE_SIZE = 200

st.set_page_config(
    page_title='Price Deconstruction',
    layout='wide',
    page_icon=':rocket:'
)

st.title('Price Deconstruction')

underlying_function_tab, deconstruction_tab = st.tabs(
    ['Define the underlying function', 'Define Construction technique'])

with underlying_function_tab:
    st.text_input(
        'Define the underlying hidden function of y with respect to x',
        value='x**2 + 2x+sin(x)',
        key='x_distribution_input',
        help='Define the underlying hidden relationship between y and x.\n\n'
             'In the formula Y=f(X) + e, this defines f(X)'
    )

    parsed_function = sp.parsing.parse_expr(st.session_state.x_distribution_input, transformations='all')
    compiled_func = sp.lambdify(sp.abc.x, expr=parsed_function)
    st.write(parsed_function)

    x_low, x_high = st.slider(
        'Pick the range of x',
        min_value=-100.,
        max_value=100.,
        value=(-50., 50.)
    )

    X_range = np.linspace(x_low, x_high, 10_000)
    X = X_range.reshape(-1, 1)
    y = compiled_func(X_range)

    fig, ax = plt.subplots()
    ax.plot(X_range, y)
    ax.set_ylabel('Hidden function')
    ax.set_xlabel('X')

    _, _, col, _, _ = st.columns([1,1,2,1,1])
    with col:
        st.pyplot(fig)

with deconstruction_tab:
    model_selected = st.selectbox(
        'Select the model we will use',
        ALL_MODELS.keys()
    )

    hyper_parameter_column, model_plot_cols = st.columns(2)

    streamlit_model = ALL_MODELS[model_selected]
    with hyper_parameter_column:
        st.subheader('Hyper Parameter Selections')
        streamlit_model.render_hyper_parameters()

    streamlit_model.fit(X, y)

    predictions = streamlit_model.predict(X)

    with model_plot_cols:
        st.subheader('Plotting')
        fig, ax = plt.subplots()

        plot_preds = ax.plot(X_range, predictions, label=f'Prediction from {model_selected}')
        plot_actuals = ax.plot(X_range, y, label='Actual Data')

        show_residuals = st.checkbox('Show residuals?', value=False)
        show_residuals_squared = st.checkbox('Show residuals squared', value=False)

        if show_residuals:
            ax_res = ax.twinx()
            ax_res.set_ylabel("Residuals")
            ax_res.plot(X_range, y - predictions, label='Residuals', color='Green')
        if show_residuals_squared:
            ax2 = ax.twinx()
            ax2.set_ylabel("Residual Squared")
            ax2.plot(X_range, (y - predictions) ** 2, label='Residuals^2', color='Red')
            if show_residuals:
                ax2.spines['right'].set_position(('outward', 60))

        ax.legend()

        st.pyplot(fig)

        fig, ax = plt.subplots()

        if model_selected == 'GBM':
            streamlit_model.plot_trees()
