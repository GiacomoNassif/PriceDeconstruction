from abc import ABC, abstractmethod, abstractproperty

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.tree import plot_tree


class StreamlitModel(ABC):
    def __init__(self, model: BaseEstimator):
        self.model = model
        self.hyper_parameters = {}

    @abstractmethod
    def render_hyper_parameters(self):
        pass

    def fit(self, X, y):
        self.model.set_params(**self.hyper_parameters)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class StreamlitLM(StreamlitModel):

    def __init__(self):
        super().__init__(LinearRegression())

    def render_hyper_parameters(self):
        pass


class StreamlitGBM(StreamlitModel):

    def __init__(self):
        self.model = GradientBoostingRegressor()
        super().__init__(self.model)
        self.depth_of_trees = None
        self.number_of_trees = None

    def render_hyper_parameters(self):
        self.hyper_parameters['n_estimators'] = st.slider('Number of trees', min_value=1, max_value=10, step=1)
        self.hyper_parameters['max_depth'] = st.slider('Depth of trees', min_value=1, max_value=10, step=1)
        self.hyper_parameters['learning_rate'] = st.slider('Learning Rate', min_value=0.01, max_value=1., step=0.05,
                                                           value=1.)

    def plot_trees(self):
        if self.hyper_parameters['max_depth'] >= 4:
            st.warning('Too much depth on the trees to plot reasonably. Please pick Depth <= 3')
            return

        number_of_trees = self.hyper_parameters['n_estimators']
        fig, ax = plt.subplots()

        plot_this_tree = st.selectbox('Pick tree to plot', [i + 1 for i in range(number_of_trees)])
        plot_tree(self.model.estimators_[plot_this_tree - 1][0], ax=ax)
        st.pyplot(fig)


ALL_MODELS: dict[str, StreamlitModel] = {
    'GBM': StreamlitGBM(),
    'LM': StreamlitLM()
}
