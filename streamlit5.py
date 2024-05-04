
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
import seaborn as sns
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_curve, auc, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures

# Page layout
st.set_page_config(page_title='The Machine Learning Hyperparameter Optimization App',
    layout='wide')

st.write("""
# The Machine Learning Hyperparameter Optimization App
**(Regression Edition)**

In this implementation, the *RandomForestRegressor()* function is used in this app for building a regression model using the **Random Forest** algorithm.
""")

# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
st.sidebar.header('Set Parameters')
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

st.sidebar.subheader('Learning Parameters')
parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, 10, 50)
parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
st.sidebar.write('---')

parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, 1)
st.sidebar.number_input('Step size for max_features', 1)
st.sidebar.write('---')

parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

st.sidebar.subheader('General Parameters')
parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

# Parameter grid
n_estimators_range = np.arange(parameter_n_estimators, parameter_n_estimators + parameter_n_estimators_step + 1, parameter_n_estimators_step)
max_features_range = np.arange(1, parameter_max_features + 1, 1)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

# Displays the dataset
st.subheader('Dataset')

# Model building
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href

def build_model(df):
    X = df.iloc[:, :-1] 
    Y = df.iloc[:, -1] 
    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size/100, random_state=42) # Ensure consistent train-test split

    # Check if the training set has enough samples for cross-validation
    if len(X_train) < 3: 
        st.warning("Training set has too few samples for cross-validation.")
        return

    # Random Forest Regression
    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
                                random_state=parameter_random_state,
                                max_features=parameter_max_features,
                                criterion='squared_error', 
                                min_samples_split=parameter_min_samples_split,
                                min_samples_leaf=parameter_min_samples_leaf,
                                bootstrap=parameter_bootstrap,
                                oob_score=parameter_oob_score,
                                n_jobs=parameter_n_jobs)

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=parameter_n_estimators,
                                random_state=parameter_random_state,
                                max_features=parameter_max_features,
                                criterion='gini',  
                                min_samples_split=parameter_min_samples_split,
                                min_samples_leaf=parameter_min_samples_leaf,
                                bootstrap=parameter_bootstrap,
                                oob_score=parameter_oob_score,
                                n_jobs=parameter_n_jobs)
    # Perform grid search
    param_grid = dict(max_features=range(1, parameter_max_features + 1),
                      n_estimators=range(parameter_n_estimators, parameter_n_estimators + parameter_n_estimators_step + 1, parameter_n_estimators_step))
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3) 
    grid.fit(X_train, Y_train)

    # Linear Regression
    lr = LinearRegression()

    # Logistic Regression
    logistic_regression = LogisticRegression()

    # Lasso Regression
    lasso = Lasso(alpha=1.0, random_state=parameter_random_state)

    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier()

    # Support Vector Machine Classifier
    svm_classifier = SVC()

    # Polynomial Regression
    poly_reg = PolynomialFeatures(degree=2)

    # Model selection
    models = {
        "Random Forest Regression": rf,
        "Linear Regression": lr,
        "Polynomial Regression": poly_reg,
        "Logistic Regression": logistic_regression,
        "Lasso Regression": lasso,
        "RandomForest Classifier":rf_classifier,
        "Decision Tree Classifier": dt_classifier,
        "Support Vector Machine Classifier": svm_classifier
    }

    selected_model = st.selectbox("Select model", list(models.keys()))

    model = models[selected_model]

    # Train logistic regression model
    if selected_model == "Logistic Regression":
        Y_train_binary = (Y_train > Y_train.mean()).astype(int)
        model.fit(X_train, Y_train_binary)

    # Model training
    elif selected_model == "Lasso Regression":
        model.fit(X_train, Y_train)

    

    else:
        model.fit(X_train, Y_train)
    if selected_model=="Polynomial Regression":
        X_test_poly = poly_reg.transform(X_test)
        Y_pred_test = model.predict(X_test_poly)
    else:

        Y_pred_test = model.predict(X_test)

        st.subheader('Model Performance')

    if selected_model == "Logistic Regression":
        st.write('Accuracy:')
        st.info(accuracy_score((Y_test > Y_train.mean()).astype(int), Y_pred_test))
    elif selected_model == "Lasso Regression":
        st.write('Coefficient of determination ($R^2$):')
        st.info(r2_score(Y_test, Y_pred_test))

        st.write('Error (MSE or MAE):')
        st.info(mean_squared_error(Y_test, Y_pred_test))

    elif selected_model=="RandomForest Classifier"or selected_model=="Decision Tree Classifier" or selected_model=="Support Vector Machine Classifier" or selected_model=="Polynomial Regression":
        st.write('Accuracy:')
        accuracy = accuracy_score(Y_test, Y_pred_test)
        st.write('Classification Report:')
        classification_rep = classification_report(Y_test, Y_pred_test, output_dict=True)
        st.table(pd.DataFrame(classification_rep).transpose())

    else:
        st.write('Coefficient of determination ($R^2$):')
        st.info(r2_score(Y_test, Y_pred_test))

        st.write('Error (MSE or MAE):')
        st.info(mean_squared_error(Y_test, Y_pred_test))

        if selected_model == "Linear Regression":
            st.write('Model Coefficients:')
            st.write(model.coef_)
            st.write('Model Intercept:')
            st.write(model.intercept_)

        # Plot ROC curve and AUC score for logistic regression
        if selected_model == "Logistic Regression":
            fpr, tpr, thresholds = roc_curve((Y_test > Y_train.mean()).astype(int), Y_pred_test)
            auc_score = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig.update_layout(title='ROC Curve',
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate')
            st.plotly_chart(fig)

            st.write('Area under the Curve (AUC):')
            st.info(auc_score)

        # Process grid data
        grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]), pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])], axis=1)
        grid_contour = grid_results.groupby(['max_features', 'n_estimators']).mean()
        grid_reset = grid_contour.reset_index()
        grid_reset.columns = ['max_features', 'n_estimators', 'R2']
        grid_pivot = grid_reset.pivot_table(index='max_features', columns='n_estimators', values='R2')

        # # Plot heatmap
        plt.figure(figsize=(4, 2))
        sns.heatmap(grid_pivot, annot=True, cmap='viridis', fmt=".3f", linewidths=0.5)
        plt.title('Grid Search Mean Test Scores')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Max Features')
        st.pyplot()


        x = grid_pivot.columns.values
        y = grid_pivot.index.values
        z = grid_pivot.values

        layout = go.Layout(
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text='n_estimators')
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text='max_features')
            ))
        fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
        fig.update_layout(title='Hyperparameter tuning',
                          scene=dict(
                              xaxis_title='n_estimators',
                              yaxis_title='max_features',
                              zaxis_title='R2'),
                          autosize=False,
                          width=800, height=800,
                          margin=dict(l=65, r=50, b=65, t=90))
        st.plotly_chart(fig)

        # Save grid data
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        z = pd.DataFrame(z)
        df = pd.concat([x, y, z], axis=1)
        st.markdown(filedownload(grid_results), unsafe_allow_html=True)
  


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting CSV file to be uploaded.')
    if st.button('Use Example Dataset'):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat([X, Y], axis=1)
        st.markdown('Using the **Diabetes** dataset as the example.')
        st.write(df.head(5))
        build_model(df)
