import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px

from sqlalchemy import create_engine
from pymongo import MongoClient

# ------------------------------------------------
# Backend Functions
# ------------------------------------------------
def clean_data(df):
    """Basic preprocessing: handle missing values"""
    df = df.dropna()
    return df

def load_mysql_data(host, user, password, db, query="SELECT * FROM mytable"):
    """Load data from MySQL"""
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db}")
    df = pd.read_sql(query, engine)
    return df

def load_mongo_data(uri="mongodb://localhost:27017/", db="testdb", collection="mycollection"):
    """Load data from MongoDB"""
    client = MongoClient(uri)
    db = client[db]
    coll = db[collection]
    df = pd.DataFrame(list(coll.find()))
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df

def train_model(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    with open("model.pkl", "wb") as f:
        pickle.dump((scaler, model), f)

    return acc

def predict_input(input_dict):
    with open("model.pkl", "rb") as f:
        scaler, model = pickle.load(f)

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return prediction

# ------------------------------------------------
# Frontend (Dash)
# ------------------------------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("ML Data Preprocessing + Prediction App"),

    # Choose data source
    dcc.RadioItems(
        id="data-source",
        options=[
            {"label": "Upload CSV", "value": "csv"},
            {"label": "Load from MySQL", "value": "mysql"},
            {"label": "Load from MongoDB", "value": "mongodb"}
        ],
        value="csv",
        labelStyle={"display": "inline-block", "margin-right": "15px"}
    ),

    # CSV upload
    dcc.Upload(
        id='upload-data',
        children=html.Div(['📂 Drag and Drop or ', html.A('Select CSV File')]),
        style={
            'width': '60%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),

    # MySQL inputs
    html.Div([
        html.Label("MySQL Host"), dcc.Input(id="mysql-host", type="text", value="localhost"),
        html.Label("User"), dcc.Input(id="mysql-user", type="text", value="root"),
        html.Label("Password"), dcc.Input(id="mysql-pass", type="password"),
        html.Label("Database"), dcc.Input(id="mysql-db", type="text"),
        html.Label("Query"), dcc.Input(id="mysql-query", type="text", value="SELECT * FROM mytable"),
        html.Button("Load MySQL Data", id="load-mysql-btn", n_clicks=0)
    ], id="mysql-inputs", style={"display": "none"}),

    # MongoDB inputs
    html.Div([
        html.Label("Mongo URI"), dcc.Input(id="mongo-uri", type="text", value="mongodb://localhost:27017/"),
        html.Label("Database"), dcc.Input(id="mongo-db", type="text", value="testdb"),
        html.Label("Collection"), dcc.Input(id="mongo-collection", type="text", value="mycollection"),
        html.Button("Load MongoDB Data", id="load-mongo-btn", n_clicks=0)
    ], id="mongo-inputs", style={"display": "none"}),

    html.Div(id='output-data-upload'),

    dcc.Dropdown(id='eda-column', placeholder="Select column for histogram"),
    dcc.Graph(id='eda-graph'),

    html.Div([
        html.Label("Select Target Column:"),
        dcc.Input(id="target-col", type="text", placeholder="Enter target column name"),
        html.Button("Train Model", id="train-btn", n_clicks=0),
        html.Div(id="train-output")
    ], style={'margin-top': '20px'}),

    html.H3("Prediction Section"),
    html.Div(id="prediction-inputs"),
    html.Button("Predict", id="predict-btn", n_clicks=0),
    html.Div(id="predict-output")
])

# ------------------------------------------------
# Callbacks
# ------------------------------------------------
global_df = pd.DataFrame()

@app.callback(
    Output("mysql-inputs", "style"),
    Output("mongo-inputs", "style"),
    Input("data-source", "value")
)
def toggle_inputs(source):
    if source == "mysql":
        return {"display": "block"}, {"display": "none"}
    elif source == "mongodb":
        return {"display": "none"}, {"display": "block"}
    else:
        return {"display": "none"}, {"display": "none"}

@app.callback(
    Output('output-data-upload', 'children'),
    Output('eda-column', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('load-mysql-btn', 'n_clicks'),
    State('mysql-host', 'value'),
    State('mysql-user', 'value'),
    State('mysql-pass', 'value'),
    State('mysql-db', 'value'),
    State('mysql-query', 'value'),
    Input('load-mongo-btn', 'n_clicks'),
    State('mongo-uri', 'value'),
    State('mongo-db', 'value'),
    State('mongo-collection', 'value'),
    State('data-source', 'value')
)
def handle_data(csv_contents, filename,
                mysql_clicks, host, user, pw, db, query,
                mongo_clicks, uri, mdb, coll,
                source):
    global global_df
    try:
        if source == "csv" and filename:
            global_df = pd.read_csv(filename)
        elif source == "mysql" and mysql_clicks > 0:
            global_df = load_mysql_data(host, user, pw, db, query)
        elif source == "mongodb" and mongo_clicks > 0:
            global_df = load_mongo_data(uri, mdb, coll)
        else:
            return "No data loaded yet.", []
        
        global_df = clean_data(global_df)
        table = dash_table.DataTable(
            data=global_df.head(10).to_dict('records'),
            columns=[{"name": i, "id": i} for i in global_df.columns],
            page_size=10
        )
        options = [{'label': col, 'value': col} for col in global_df.columns]
        return html.Div([html.H5("Preview of Data"), table]), options
    except Exception as e:
        return f"Error loading data: {str(e)}", []

@app.callback(
    Output('eda-graph', 'figure'),
    Input('eda-column', 'value')
)
def update_histogram(col):
    if col and not global_df.empty:
        return px.histogram(global_df, x=col)
    return {}

@app.callback(
    Output("train-output", "children"),
    Input("train-btn", "n_clicks"),
    State("target-col", "value")
)
def train_callback(n_clicks, target_col):
    if n_clicks > 0 and target_col in global_df.columns:
        acc = train_model(global_df, target_col)
        return f"✅ Model trained! Accuracy: {acc:.2f}"
    return ""

@app.callback(
    Output("prediction-inputs", "children"),
    Input("train-output", "children")
)
def generate_prediction_inputs(train_msg):
    if "✅" in train_msg:
        inputs = []
        for col in global_df.columns[:-1]:
            inputs.append(html.Div([
                html.Label(col),
                dcc.Input(id=f"input-{col}", type="number", placeholder=f"Enter {col}")
            ]))
        return inputs
    return []

@app.callback(
    Output("predict-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(f"input-{col}", "value") for col in global_df.columns[:-1]]
)
def predict_callback(n_clicks, *values):
    if n_clicks > 0:
        input_dict = {col: val for col, val in zip(global_df.columns[:-1], values)}
        pred = predict_input(input_dict)
        return f"🎯 Predicted Class: {pred}"
    return ""

# ------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
