import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load and preprocess dataset
df = pd.read_csv("C:\\Users\\Sanjali\\Downloads\\dss-demographics-2021-sa2-december-2024.csv")
df = df.dropna()
df['high_jobseeker'] = (df['JobSeeker Payment'] > df['JobSeeker Payment'].median()).astype(int)

features = ['Age Pension', 'Disability Support Pension', 'Carer Payment']
scaler = StandardScaler()

# Initialize Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Logistic Regression Dashboard", style={"textAlign": "center"}),

    html.Div([
        dcc.Dropdown(
            id='feature-selector',
            options=[{'label': col, 'value': col} for col in features],
            value=features,
            multi=True,
            placeholder="Select Features for Analysis"
        ),
    ], style={"width": "60%", "margin": "auto"}),

    html.Div(id='summary-metrics', style={"margin": "30px", "textAlign": "center"}),

    dcc.Tabs([
        dcc.Tab(label='Model Performance', children=[
            dcc.Graph(id='roc-curve'),
            dcc.Graph(id='conf-matrix')
        ]),
        dcc.Tab(label='Model Insights', children=[
            dcc.Graph(id='feature-importance-heatmap'),
            dcc.Graph(id='scatter-plot'),
            dcc.Graph(id='prediction-bar'),
            html.H3("Prediction Results Table"),
            dash_table.DataTable(
                id='results-table',
                columns=[{"name": col, "id": col} for col in features] +
                        [{"name": "Actual", "id": "Actual"},
                         {"name": "Predicted", "id": "Predicted"}],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center"}
            )
        ])
    ])
])

# Callback to update visuals
@app.callback(
    Output('roc-curve', 'figure'),
    Output('feature-importance-heatmap', 'figure'),
    Output('conf-matrix', 'figure'),
    Output('scatter-plot', 'figure'),
    Output('prediction-bar', 'figure'),
    Output('results-table', 'data'),
    Output('summary-metrics', 'children'),
    Input('feature-selector', 'value')
)
def update_dashboard(selected_features):
    try:
        if not selected_features:
            selected_features = features

        X_sel = df[selected_features]
        y = df['high_jobseeker']
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_scores = model.predict_proba(X_test_scaled)[:, 1]
        y_preds = (y_scores > 0.5).astype(int)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        roc_fig.update_layout(title=f"ROC Curve (AUC = {roc_auc:.2f})",
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate')

        # Feature Importance Heatmap
        coeff_data = pd.DataFrame({
            "Feature": selected_features,
            "Coefficient": model.coef_[0]
        })
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=[coeff_data["Coefficient"].values],
            x=coeff_data["Feature"],
            y=["Coefficient"],
            colorscale='RdBu',
            zmid=0
        ))
        heatmap_fig.update_layout(
            title="Feature Importance Heatmap",
            xaxis_title="Features",
            yaxis_title="",
            height=300
        )

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_preds)
        conf_fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Low', 'Predicted High'],
            y=['Actual Low', 'Actual High'],
            colorscale='Viridis'))
        conf_fig.update_layout(title='Confusion Matrix')

        # Scatter Plot
        if len(selected_features) >= 2:
            scatter_fig = px.scatter(X_test, x=selected_features[0], y=selected_features[1],
                                     color=y_preds.astype(str),
                                     title="Scatter Plot of Predictions",
                                     labels={"color": "Predicted"})
        else:
            scatter_fig = go.Figure()
            scatter_fig.update_layout(title="Select at least 2 features for Scatter Plot")

        # Prediction Bar
        result_df = X_test.copy()
        result_df["Actual"] = y_test.values
        result_df["Predicted"] = y_preds
        pred_count = result_df['Predicted'].value_counts().sort_index()
        bar_fig = px.bar(
            x=['Low', 'High'], y=pred_count.values,
            labels={'x': 'Prediction', 'y': 'Count'},
            title='Prediction Distribution',
            color=['Low', 'High']
        )

        # Summary metrics
        top_feature = selected_features[np.argmax(np.abs(model.coef_[0]))]
        summary = html.Div([
            html.H4(f"Model AUC Score: {roc_auc:.2f}"),
            html.H4(f"Most Influential Feature: {top_feature}")
        ])

        return roc_fig, heatmap_fig, conf_fig, scatter_fig, bar_fig, result_df.to_dict('records'), summary

    except Exception as e:
        print("Callback error:", e)
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], html.Div()

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
