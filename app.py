#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import shap
import plotly.graph_objects as go


# In[2]:


fontawesome_stylesheet = "https://use.fontawesome.com/releases/v5.8.1/css/all.css"
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, fontawesome_stylesheet])
server = app.server


# In[3]:


df = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/candy-power-ranking/candy-data.csv', sep=',')
df.competitorname = df.competitorname.str.replace('Õ','')
df.head()


# In[4]:


feature_dict = {c:i for i,c in enumerate(df.columns.values)}


# In[5]:


model = load('random_forest.joblib') 


# In[6]:


checklist_ingredients = dcc.Checklist(
    id='checklist_ingredients',
    options=[
        {'label': 'Chocolate', 'value': 'chocolate'},
        {'label': 'Fruity', 'value': 'fruity'},
        {'label': 'Caramel', 'value': 'caramel'},
        {'label': 'Peanuty/Almondy', 'value': 'peanutyalmondy'},
        {'label': 'Nougat', 'value': 'nougat'},
        {'label': 'Crisped Rice Wafer', 'value': 'crispedricewafer'},
    ],
    value=['chocolate', 'caramel'],
    labelStyle={'display': 'inline-block', 'margin-left':'10px'}
)


# In[7]:


checklist_props = dcc.Checklist(
    id='checklist_props',
    options=[
        {'label': 'Hard', 'value': 'hard'},
        {'label': 'Bar', 'value': 'bar'},
        {'label': 'Pluribus', 'value': 'pluribus'},
    ],
    value=['bar'],
    labelStyle={'display': 'inline-block', 'margin-left':'10px'},
)


# In[8]:


slider_price = dcc.Slider(
    id='slider_price',
    min=0.0,
    max=1.0,
    value=0.5,
    marks={
        0.0: {'label': '0%'},
        0.25: {'label': '25%'},
        0.5: {'label': '50%'},
        0.75: {'label': '75%'},
        1.0 : {'label': '100%'}
    },
    tooltip={"placement": "bottom", "always_visible": True},
)


# In[9]:


slider_sugar = dcc.Slider(
    id='slider_sugar',
    min=0.0,
    max=1.0,
    value=0.5,
    marks={
        0.0: {'label': '0%'},
        0.25: {'label': '25%'},
        0.5: {'label': '50%'},
        0.75: {'label': '75%'},
        1.0 : {'label': '100%'}
    },
    tooltip={"placement": "bottom", "always_visible": True},
)


# In[10]:


refresh_button = html.Div(children=[
    html.Button('Submit', id='submit-val', n_clicks=0),
])


# In[11]:


dropdown_options = [{'label': c, 'value': c} for i,c in enumerate(df.competitorname)]

dropdown = dcc.Dropdown(
    id='dropdown',
    options=dropdown_options,
    value='100 Grand'
    )


# In[12]:


@app.callback(
    Output("slider_price", "value"),
    Output("slider_sugar", "value"),
    Output("checklist_ingredients", "value"),
    Output("checklist_props", "value"),
    Input("dropdown", "value"))
def change_values(competitor):
    row = df.loc[df.competitorname == competitor]
    price = row['pricepercent'].values[0]
    sugar = row['sugarpercent'].values[0]
    ingredients = [c for c in row.columns[1:7] if row[c].values == 1]
    props = [c for c in row.columns[7:10] if row[c].values == 1]
    
    
    return price, sugar, ingredients, props


# In[13]:


app.layout = html.Div([
    html.Div([html.H1('Candy popularity predictor')], style={'padding': '25px'}),
    
    ## left panel
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H5('Inspect a competitor'),
                html.Div([dropdown]),
            ])
        ]),
        html.Div([], style={'width': '5px', 'display': 'inline-block', 'vertical-align': 'top'}),
        dbc.Card([
            dbc.CardBody([
                html.H5('Ingredients'),
                checklist_ingredients,
                html.Div([], style={'height': '15px'}),
                
                html.H5('Other properties'),
                checklist_props,
                html.Div([], style={'height': '15px'}),
                
                html.H5('Price percentile'),
                slider_price,
                html.Div([], style={'height': '15px'}),
                
                html.H5('Sugar percentile'),
                slider_sugar,
            ])
        ])
   ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '25px'}),
    
    ## Divider
    html.Div([], style={'width': '2%', 'display': 'inline-block', 'vertical-align': 'top'}),
    
    
    ## right panel
    html.Div([
        dbc.Card([
            dbc.CardBody([   
                html.Div('Prediction'),
                html.Div(id="prediction_text"),
                html.Div('/ 100')
            ]),
        ], style={"text-align": "center"}),
        html.Div([], style={'width': '5px', 'display': 'inline-block', 'vertical-align': 'top'}),
        dbc.Card([
            dbc.CardBody([
                html.H5('Analysis'),
                html.Div(id="prediction"),
            ]),
        ]),
        
    ], style={'width': '64%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '25px'}),
])


# In[14]:


class WaterfallData():
    def __init__ (self, shap_test):
        self.values = shap_test[0].values
        self.base_values = shap_test[0].base_values[0]
        self.data = shap_test[0].data
        self.feature_names = shap_test.feature_names


# In[15]:


def figure_to_html_img(figure):
    """ figure to html base64 png image """ 
    try:
        tmpfile = io.BytesIO()
        figure.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        shap_html = html.Img(src=f"data:image/png;base64, {encoded}")
        return shap_html
    except AttributeError:
        return ""


# In[16]:


@app.callback(
    Output("prediction", "children"),
    Output('prediction_text', 'children'),
    Input('slider_price', "value"),
    Input('slider_sugar', 'value'),
    Input('checklist_ingredients', 'value'),
    Input('checklist_props', 'value'),
)
def get_prediction(price, sugar, ingredients, props):
    
    X = np.repeat(0,9)
    
    for f in ingredients:
        X[feature_dict[f]-1] = 1

    for f in props:
        X[feature_dict[f]-1] = 1

    X = np.append(X, sugar)  
    X = np.append(X, price)
    X = pd.DataFrame(X.reshape(1, -1), columns=df.drop(columns=['competitorname', 'winpercent']).columns.values)
    
    y = model.predict(X)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        base=shap_values.base_values[0][0],
        measure = ['relative', 'relative', 'relative', 'relative', 'relative', 'relative','relative', 'relative','relative', 'relative', 'relative', 'total'],
        x = ['Chocolate', 'Fruity', 'Caramel', 'Peanuty/Almondy', 'Nougat',
           'Crisped Ricewafer', 'Hard', 'Bar', 'Pluribus', 'Sugar %',
           'Price %'],
        textposition = "outside",
        text = np.around(shap_values.data[0], decimals=2),
        y = shap_values.values[0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))

    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    
    text = html.H2('{:.2f}'.format(y[0]))
     
    return dcc.Graph(figure=fig), text


# In[17]:


if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)

