import pandas as pd
from dash import Dash, html, dcc, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.io as pio
import altair as alt
import os
import base64
import io
import datetime
import string


import openai as ai
import time

import warnings
warnings.filterwarnings("ignore")




############################### App Set Up ###################################

# ai.api_key = "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"

api_key = dcc.Input(
    id="api-key",
    type='text',
    placeholder="Paste an API Key",
    style={"border-radius": "7px", "width":"90%", "margin-left":"15px"}
)


# GPT FUNCTION
def generate_gpt3_response(user_text, print_output=False):
    """
    Query OpenAI GPT-3 for the specific key and get back a response
    :type user_text: str the user's text to query for
    :type print_output: boolean whether or not to print the raw output JSON
    """
    time.sleep(1)
    completions = ai.Completion.create(
        engine='text-davinci-003',  # Determines the quality, speed, and cost.
        temperature=0.5,            # Level of creativity in the response
        prompt=user_text,           # What the user typed in
        max_tokens=200,             # Maximum tokens in the prompt AND response
        n=1,                        # The number of completions to generate
        stop=None,                  # An optional setting to control response generation
    )

    # Displaying the output can be helpful if things go wrong
    if print_output:
        print(completions)

    # Return the first choice's text
    return completions.choices[0].text



# Helper function to read in uploaded data
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('ISO-8859-1')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    
    # Drop na - TODO: more data validation
    df = df.dropna()

    # Take a sample
    sample = df.head(35)

    # Run GPT topic
    sample['GPT-text-davinci-003'] = sample['understanding_comments'].apply(lambda x: \
              generate_gpt3_response\
              ("I am giving you text from an employee survey, tell me the general high level topic as Culture, Nature of Job, Manager, Leadership, \
                Compensation, Stress, Burnout, Location, Safety, Inclusion, Relationships, Career Advancement, Learning and Development, Work Life Balence, Satisfaction, or Other '{}'. ".format(x)))
    
    # Run GPT sentiment
    sample['GPT-text-davinci-003-sentiment'] = sample['understanding_comments'].apply(lambda x: \
              generate_gpt3_response\
              ("I am giving you text from an employee survey, tell me the sentiment of the text as positive, negative, or mixed '{}'. ".format(x)))
    

    # Formatting
    sample['GPT-text-davinci-003'] = sample['GPT-text-davinci-003'].apply(lambda x: (x.replace('\n','').replace('\r','') ))
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].apply(lambda x: (x.replace('\n','').replace('\r','')))  
    sample['GPT-text-davinci-003'] = sample['GPT-text-davinci-003'].apply(lambda x: string.capwords(x.lower().replace('.','')))

    # Get specific columns
    sample = sample[['understanding_comments', 'GPT-text-davinci-003', 'GPT-text-davinci-003-sentiment']]

    # Remove periods sometimes returned by GPT
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'.', '')
    sample['GPT-text-davinci-003'] = sample['GPT-text-davinci-003'].str.replace(r'.', '')

    # Replace common incorrect sentiments returned by GPT
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of the text is positive', 'Positive')
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of the text is positive.', 'Positive')
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of the text is negative', 'Negative')
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of the text is negative.', 'Negative')
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of the text is mixed', 'Mixed')
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of the text is mixed.', 'Mixed')



    # Rename columns
    sample = sample.rename(columns={'understanding_comments': "Respondent Comment", 'GPT-text-davinci-003': "GPT Topic", 'GPT-text-davinci-003-sentiment': "GPT Sentiment"})



    return sample.to_dict('records')





############################### Side Bar ###################################
# Side bar of the dashboard containing basic information about the dashboard and the match selector
sidebar = dbc.Col(
    dbc.Row(
        [
            html.Br(),
            html.P(" "),
            html.P(" "),
            html.P(" "),
            html.H3(
                "Topic Modeling Dashboard",
                style={
                    "font": "Helvetica",
                    "font-size": "35px",
                    "text-align": "center",
                    "color": "white",
                },
            ),
            html.P(" "),
            html.Br(),
            html.Br(),
            api_key,
            html.Button('Submit', id='submit-val', n_clicks=0, style={"border-radius": "7px", "width":"90%", "margin-left":"15px"}),
            html.Br(),
            html.Br(),
            html.P(""),
            html.P(""),
            html.P(
                "This dashboard utilizes GPT-3.5 (text-davinci-003) API from OpenAI to conduct topic modeling on survey comments.",
                style={"text-align": "left", "color": "white"},
            ),
            html.Hr(),
            html.Br(),
            html.P(
                "Upload a .csv file with 1 column called 'understanding_comments'. This may take a few minutes to return the topics.",
                style={"text-align": "left", "color": "white"},
            ),
            html.Br(),
            html.Br()
        ],
    ),
    width=2,
    style={
        "border-width": "0",
        "backgroundColor": "#4a387fe3",
    },
)

################################### Tabs #####################################


topic_tab = html.Div(
    [
        
        # DATA INPUT
        dbc.Row(
            [
                html.P(" "),
                html.B(
                    [
                        "Data Input",
                        html.Span(
                            "(?)",
                            id="tooltip-stat",
                            style={
                                "textDecoration": "underline",
                                "cursor": "pointer",
                                "font-size": "10px",
                                "vertical-align": "top",
                            },
                        ),
                    ],
                    style={"font-size": "30px"},
                ),
                dbc.Tooltip(
                    "Upload survey comments in one .csv file here. It should contain only 1 column called 'understanding_comments'",
                    target="tooltip-stat",
                ),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '200px',
                        'lineHeight': '200px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                ),
                dcc.Store(id="data_storage"),
                dcc.Store(id="api_storage")



            ]
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        # OUTPUT TABLE
        dbc.Row(
            [
                html.B("Comments Table", style={"font-size": "30px"}),
                html.P(
                    "The table below shows uploaded comments and their topic assigned by GPT.",
                ),
                html.Div(
                    id="topic_table",
                    style={
                        "display": "block",
                        "overflow": "scroll",
                        "width": "100%",
                        "height": "500px",
                    },
                ),
            ]
        ),

        html.Br(),
        html.Br(),
        html.Br(),
        # PLOT OUTPUT
        dbc.Row([
            dbc.Col([ 
                html.B("Summary Plot", style={"font-size": "30px"}),
                html.P(
                        "The plot below shows counts of topics from the table above.",
                    ),
                html.Iframe(id="graph", style={
                        "display": "block",
                        "overflow": "scroll",
                        "width": "100%",
                        "height": "1000px",
                    })
            ]),
            dbc.Col([ 
                html.B("Sentiment Plot", style={"font-size": "30px"}),
                html.P(
                        "The plot below shows counts of sentiments from the table above.",
                    ),
                html.Iframe(id="graph2", style={
                        "display": "block",
                        "overflow": "scroll",
                        "width": "100%",
                        "height": "1000px",
                    })
            ]),


        ])
    ]
)


################################ App Setup ################################
# Basic setup of the dashboard app
app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
app.config["suppress_callback_exceptions"] = True

app.title = "Topic Modeling Dashboard"
server = app.server


############################## App Layout #################################
# Container that organizes the layout of the dashboard
# Main components of the layout are the side bar and 
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                sidebar,
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            topic_tab,
                                            label="Topic Tab",
                                            tab_id="topic-tab",
                                            tab_style={"color": "black"},
                                            active_label_style={"color": "#4a387fe3"},
                                        )
                                    ]
                                ),
                            ]
                        )
                    ],
                    width=10,
                ),
            ]
        ),
        dbc.Row(
            [
                dcc.Markdown(
                    "This Dashboard was created and maintained by Adam Morphy. "
                )
            ],
            style={
                "height": "60px",
                "background-color": "#e5e5e5",
                "font-size": "14px",
                "padding-left": "20px",
                "padding-top": "20px",
            },
        ),
    ],
    fluid=True,
)

################################ Functions ###################################
# List of functions that are used to support the dashboard 
# Functions are organized according to each of the two tabs



#### Set API Key ####
@app.callback(
    Output("api_storage", "data"),
    [
        Input('submit-val', 'n_clicks'),
        State('api-key', 'value')
    ]
)
def set_key(n_clicks, value):
    """
    Set API key tp openai
    """
    return value





#### Topic Modeling Tab ####


# On upload, run gpt and update table storage
@app.callback(Output("data_storage", "data"),
              Input("api_storage", "data"),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(api, list_of_contents, list_of_names, list_of_dates):
    
    if api == "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":
        ai.api_key = os.getenv('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')


    if list_of_contents is not None:
        output_data = parse_contents(list_of_contents, list_of_names, list_of_dates)

        return output_data


# Results table
@app.callback(
    Output("topic_table", component_property="children"),
    [
        Input("data_storage", "data"),
        Input('submit-val', 'n_clicks')
    ],
)
def plot_table(data, _):
    """
    Returns the passing summary statistics table

    Parameters
    ----------
    match_id : int
        the match id selected by the match selector

    Returns
    -------
    chart : the summary table 
    """
    data_table = pd.DataFrame(data)

    dt_columns = [
        {"name": i, "id": i, "deletable": True, "selectable": True}
        for i in data_table.columns
    ]

    chart = dash_table.DataTable(
        id="table_1_chart",
        columns=dt_columns,
        data=data_table.to_dict("records"),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable=False,
        row_selectable="multi",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current=0,
        page_size=10,
        style_data={"whiteSpace": "normal", "height": "auto"},
        style_as_list_view=True,
        export_format="csv",
        style_cell={
            "font_family": "Arial",
            #'font_size': '26px',
            "text_align": "left",
        },
        style_header={
            "backgroundColor": "#4a387fe3",
            "color": "white",
            "fontWeight": "bold",
            "text_align": "left",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Respondent Comment"}, "width": "75%"},
            {"if": {"column_id": "GPT Topic"}, "width": "10%"},
            {"if": {"column_id": "GPT Sentiment"}, "width": "10%"}
        ],
    )

    return chart


# Column highlighter
@app.callback(
    Output("table_1_chart", "style_data_conditional"),
    Input("table_1_chart", "selected_columns"),
    
)
def update_styles(selected_columns):
    """
    Add highlighter function for the statistics table.
    Selected columns will be highlighted with the specified background color.
    """
    return [
        {"if": {"column_id": i}, "background_color": "#9AC5E8"}
        for i in selected_columns
    ]


# Summary plot
@app.callback(
    Output("graph", "srcDoc"),
    [
        Input("table_1_chart", "derived_virtual_data"),
        Input("table_1_chart", "derived_virtual_selected_rows")
    ], 
    prevent_initial_call=True
)
def topic_plot(df, derived_virtual_selected_rows):

    df = pd.DataFrame(df)

    # Prevents error in start up
    if ("GPT Topic" not in df.columns):
        df["GPT Topic"] = ['Topic 1', 'Topic 2', 'Topic 3']
        df["GPT Sentiment"] = ['Negative', 'Mixed', 'Positive']
    
    #df = df.groupby(by= ['GPT Topic', 'GPT Sentiment'], dropna=True).size().rename('Count').reset_index()
    df = df.dropna()

    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []



    chart = alt.Chart(df).mark_bar().encode(
        y = alt.Y("GPT Topic:N", sort='-x'),
        x = alt.X("count():Q", axis=alt.Axis(tickMinStep=1)),
        color = alt.Color('GPT Sentiment', scale = alt.Scale(domain=['Negative', 'Mixed', 'Positive'], 
                                                             range=['#ff4c3f', '#ffd351', '#b6d7a8'])
                                                             ),
        tooltip = 'GPT Sentiment'
    ).interactive().properties(height = 550,
                               width = 450).configure_axis(grid=False, domain=False)

    
    return chart.to_html()

# Sentiment plot
@app.callback(
    Output("graph2", "srcDoc"),
    [
        Input("table_1_chart", "derived_virtual_data"),
        Input("table_1_chart", "derived_virtual_selected_rows")
    ], 
    prevent_initial_call=True
)
def sentiment_plot(df, derived_virtual_selected_rows):


    df = pd.DataFrame(df)

    # Prevents error in start up
    if ("GPT Topic" not in df.columns):
        df["GPT Topic"] = ['Topic 1', 'Topic 2', 'Topic 3']
        df["GPT Sentiment"] = ['Negative', 'Mixed', 'Positive']
    
    #df = df.groupby(by= ['GPT Topic', 'GPT Sentiment'], dropna=True).size().rename('Count').reset_index()
    df = df.dropna()

    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []



    chart = alt.Chart(df).mark_bar().encode(
        y = alt.Y("GPT Sentiment:N", sort='-x'),
        x = alt.X("count():Q", axis=alt.Axis(tickMinStep=1)),
        color = alt.Color('GPT Sentiment', scale = alt.Scale(domain=['Negative', 'Mixed', 'Positive'], 
                                                             range=['#ff4c3f', '#ffd351', '#b6d7a8'])
                                                             ),
        tooltip = 'GPT Sentiment'
    ).interactive().properties(height = 550,
                               width = 450).configure_axis(grid=False, domain=False)

    
    return chart.to_html()

if __name__ == "__main__":
    app.run_server(debug=True)
