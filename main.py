import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import plot_tree
import numpy as np
import sympy as sp
from sympy.abc import x

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
    y = compiled_func(X_range)

    fig, ax = plt.subplots()

    ax.plot(X_range, y)
    st.pyplot(fig)

with deconstruction_tab:
    model_selected = st.selectbox(
        'Select the model we will use',
        ['GBM']
    )

    if model_selected == 'GBM':
        number_of_trees = st.slider('Number of trees', min_value=1, max_value=10, step=1)
        depth_of_trees = st.slider('Depth of trees', min_value=1, max_value=10, step=1)
        model = GradientBoostingRegressor(n_estimators=number_of_trees, max_depth=depth_of_trees, learning_rate=1)
        model.fit(X_range.reshape(-1, 1), y)

    preds = model.predict(X_range.reshape(-1, 1))

    fig, ax = plt.subplots()

    ax.plot(X_range, preds, label=f'Prediction from {model_selected}')
    ax.plot(X_range, y, label='Actual Data')

    show_residuals = st.checkbox('Show residuals?', value=True)
    show_residuals_squared = st.checkbox('Show residuals squared', value=False)

    if show_residuals:
        ax.plot(X_range, y - preds, label='Residuals')
    if show_residuals_squared:
        ax.plot(X_range, (y - preds) ** 2, label='Residuals^2')

    ax.legend()

    st.pyplot(fig)


    fig, ax = plt.subplots()

    plot_this_tree = st.selectbox('Pick tree to plot', [i for i in range(number_of_trees)])
    plot_tree(model.estimators_[plot_this_tree-1][0], ax=ax)
    st.pyplot(fig)
