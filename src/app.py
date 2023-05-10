import pandas as pd
import numpy as np
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
import re



import openai as ai
import tiktoken
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
def generate_gpt3_response(user_text, print_output=False, max_tokens = None, get_cost = False):
    """
    Query OpenAI GPT-3 for the specific key and get back a response
    :type user_text: str the user's text to query for
    :type print_output: boolean whether or not to print the raw output JSON
    """
    time.sleep(2)
    completions = ai.Completion.create(
        engine='text-davinci-003',  # Determines the quality, speed, and cost.
        temperature=0.5,            # Level of creativity in the response
        prompt=user_text,           # What the user typed in
        max_tokens=max_tokens,             # Maximum tokens in the prompt AND response
        n=1,                        # The number of completions to generate
        #stop=None,                  # An optional setting to control response generation
    )

    # Displaying the output can be helpful if things go wrong
    if print_output:
        print(completions)

    if get_cost:
        return {"text": completions.choices[0].text,
                "used_tokens": completions.usage.total_tokens}
    else:
        # Return the first choice's text
        return completions.choices[0].text


# Helper function to calculate tokens in text
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

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
    sample = df.head(40)

    return sample.to_dict()



# Get themes function
def get_themes(sample):

    # Limit characters in each comment
    sample['understanding_comments'] = sample['understanding_comments'].apply(lambda x: x[:100])

    all_text = " ".join( sample['understanding_comments'].tolist() )

    #all_text = all_text.replace('"', '')
    #out = generate_gpt3_response\
    #            ("I am giving you text from an employee survey. Tell me no more than 10 general high level themes, \
    #             and a summary of each theme '{}'. ".format(all_text).replace('"', ''), max_tokens=3000, print_output = True)
    
    

    num_comments = len(sample['understanding_comments'].tolist())

    bucket_size = 60
    tokens = 0
    # Get overall themes from all text
    results = []
    for bucket_num in range(bucket_size):
        end_index = (bucket_num+1)*bucket_size 
        start_index = bucket_num*bucket_size 
        
        if end_index > num_comments:
            end_index = num_comments

        text = ' '.join(sample['understanding_comments'].iloc[ start_index : end_index ].tolist() )

        out = generate_gpt3_response\
                ("I am giving you text from an employee survey, tell me no more then 10 general high level themes, and a brief summary of each theme ordered by prevalence '{}'. ".format(text), 
                max_tokens=3000,
                get_cost = True)

        tokens += out['used_tokens']
        results.append(out['text'])

        if end_index >= num_comments:
            break

    results = ' '.join(results)


    

    out = generate_gpt3_response\
        ("Combine duplicate themes from this text and keep a short description of each theme, ordered by prevalence in the text '{}'. ".format(results), 
        max_tokens=3800,
        get_cost = True)
    
    tokens += out['used_tokens']


    return out['text'], tokens



# Helper function to split a list
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]


# Get themes function
def get_themes2(dictionary):

    # Unpack given dictionary
    tokens = dictionary['used_tokens']
    text = dictionary['text']

    full_string = ' '.join(text)

    prompt_tokens = num_tokens_from_string(full_string, "gpt2")
    
    # If too many tokens for 1 prompt, split the list in half and run it again with smaller lists
    if prompt_tokens  >= 3000:
        

        A, B = split_list(text)
        
        text1 = get_themes2( {"text": A,
                                'used_tokens': 0 })
        
        text2 = get_themes2( {"text": B,
                                'used_tokens': 0 })

        # Unpack results and send to next recursive call
        full_text = [text1['text']] + [text2['text']]
        full_tokens = text1['used_tokens'] + text2['used_tokens'] 

        return get_themes2( {'text': full_text,
                             'used_tokens': full_tokens + tokens})


    elif prompt_tokens < 3000:
        
        if re.search('Themes:', full_string):
            print('Summary!!!!! OF ->>>> ', full_string)
            out = generate_gpt3_response\
                    ("I am giving you 2 numbered lists of themes with descriptions. Starting with 'Themes:', summarise this into one list by combining similar themes, but keep a short description of each theme in descending order of theme frequency '{}'. ".format(full_string), 
                    max_tokens=1000,
                    get_cost = True)

            print(' ')
            print('MADE THIS ---->>>>', out['text'])
            print(' ')
            print(' ')
            print(' ')
            print(' ')
            print(' ')
            return {'text': out['text'],
                    'used_tokens': out['used_tokens'] + tokens}

        else:

            return generate_gpt3_response\
                    ("I am giving you text from an employee survey. Starting with 'Themes: ', tell me 10 or fewer general high level themes, and a brief summary of each theme, in descending order of theme frequency '{}'. ".format(full_string), 
                    max_tokens=1000,
                    get_cost = True)
    



def get_themes3(dictionary):

    out = summariser(dictionary)

    # Unpack given dictionary
    summary_tokens = out['used_tokens']
    text = out['text']

    print("Final Summary:     ", text)
    print(" ")
    print(" ")
    print(" ")


    final_themes = generate_gpt3_response\
                    ("I am giving you text from an employee survey. Starting with 'Themes: ', tell me under 10 general high level themes, and a brief summary of each theme, in descending order of theme frequency '{}'. ".format(text), 
                    max_tokens=1000,
                    get_cost = True)
    

    return {'text': final_themes['text'],
            'used_tokens': final_themes['used_tokens'] + summary_tokens,
            'summary': text}
    



# Get themes function
def summariser(dictionary):

    # Unpack given dictionary
    tokens = dictionary['used_tokens']
    text = dictionary['text']

    full_string = ' '.join(text)

    prompt_tokens = num_tokens_from_string(full_string, "gpt2")
    
    # If too many tokens for 1 prompt, split the list in half and run it again with smaller lists
    if prompt_tokens  >= 3000:
        

        A, B = split_list(text)
        
        text1 = summariser( {"text": A,
                                'used_tokens': 0 })
        
        text2 = summariser( {"text": B,
                                'used_tokens': 0 })

        # Unpack results and send to next recursive call
        full_text = [text1['text']] + [text2['text']]
        full_tokens = text1['used_tokens'] + text2['used_tokens'] 

        return summariser( {'text': full_text,
                             'used_tokens': full_tokens + tokens})


    # If tokens is small enough to be run, summarise it!
    elif prompt_tokens < 3000:
        
        out = generate_gpt3_response\
                    ("I am giving you survey responses written by employees. Write a 50-500 word summary of this text '{}'. ".format(full_string), 
                    max_tokens=1000,
                    get_cost = True)
        
        print("SUMMARY:     ", out['text'])
        print(" ")
        print(" ")
        print(" ")

        return {'text': out['text'],
                'used_tokens': out['used_tokens'] + tokens}


        



            









# Get topics function
def get_topics(sample):

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

    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of this text is positive', 'Positive')
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of this text is positive.', 'Positive')
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of this text is negative', 'Negative')
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of this text is negative.', 'Negative')
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of this text is mixed', 'Mixed')
    sample['GPT-text-davinci-003-sentiment'] = sample['GPT-text-davinci-003-sentiment'].str.replace(r'The sentiment of this text is mixed.', 'Mixed')

    # Rename columns
    sample = sample.rename(columns={'understanding_comments': "Respondent Comment", 'GPT-text-davinci-003': "GPT Topic", 'GPT-text-davinci-003-sentiment': "GPT Sentiment"})

    return sample.to_dict()





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
                        html.A('Select Files', style={
                                                    "textDecoration": "underline",
                                                    "cursor": "pointer"
                                                })
                    ]),
                    style={
                        'width': '80%',
                        'height': '150px',
                        'lineHeight': '160px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '7px',
                        'textAlign': 'center',
                        "margin-top":"20px",
                        "margin-left":"80px"
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

        # Show if data and api is loaded correctly

        # ALL TEXT OUTPUT
        dbc.Row(
            [
                dbc.Col(
                [
    
                        html.B(
                        [
                            "GPT Theme Summary",
                            html.Span(
                                "(?)",
                                id="tooltip-stat2",
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
                        "GPT will read all of the uploaded comments and provide summary themes for them together. I have limited results to 30 themes max.",
                        target="tooltip-stat2",
                    ),
                    dbc.Button('Generate Themes', id='submit-theme', n_clicks=0, style={"border-radius": "3px", 
                                                                                            "width":"10%",
                                                                                            "height":"80%", 
                                                                                            'borderWidth': '1px',
                                                                                            "margin-left":"20px",
                                                                                            "color":"white",
                                                                                            "border-color":"black",
                                                                                            "background-color":"#4a387fe3"})
                ]),
                html.P(""),
                html.Div(id = "theme_thinking", style={"margin-left":"80px",
                                                  "color":"black",
                                                  "width":"50%"}),
                html.P(""),
                dcc.Markdown(
                    id="all_text_out",
                    style={
                        "font-size": "20px",
                        "display": "block",
                        "color": "black",
                        "border-color":"grey",
                        'borderWidth': '7px',
                        "border-style":"inset",
                        "border-radius": "1px",
                        "background-color": "#f6f6f7",
                        "width": "80%",
                        "min-height": "300px",
                        "padding-top":"20px",
                        "padding-left":"20px",
                        "margin-top":"20px",
                        "margin-left":"80px"
                    },
                ),
            ]
        ),

        html.Br(),
        html.Br(),
        html.Br(),
        # OUTPUT TABLE
        dbc.Row(
            [
                
                dbc.Col([ 
                        html.B(
                            [
                                "GPT Topic Summary",
                                html.Span(
                                    "(?)",
                                    id="tooltip-stat3",
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
                            "GPT will read all of the uploaded comments and assign a topic, and senitment for each individual comment... this may take a long time for many comments.",
                            target="tooltip-stat3",
                        ),

                        dbc.Button('Generate Topics', id='submit-topic', n_clicks=0, style={"border-radius": "3px", 
                                                                                                    "width":"10%",
                                                                                                    "height":"80%", 
                                                                                                    'borderWidth': '1px',
                                                                                                    "margin-left":"20px",
                                                                                                    "color":"white",
                                                                                                    "border-color":"black",
                                                                                                    "background-color":"#4a387fe3"})

                ]),
                html.Br(),
                html.Div(
                    id="topic_table",
                    style={
                        "display": "block",
                        "overflow": "scroll",
                        "width": "90%",
                        "height": "500px",
                        "border-radius": "3px", 


                        "border-color":"grey",
                        'borderWidth': '7px',
                        "border-style":"inset",
                        "border-radius": "1px",
                        "background-color": "#f6f6f7",
                        "padding-top":"10px",
                        "padding-left":"20px",
                        "margin-top":"20px",
                        "margin-left":"80px"
      
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
    
    if api == 'OPENAI_API_KEY':
        ai.api_key = os.getenv('OPENAI_API_KEY')

    #ai.api_key = api

    # print(os.getenv('OPENAI_API_KEY'))
        #os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        

    if list_of_contents is not None:
        output_data = parse_contents(list_of_contents, list_of_names, list_of_dates)

        return output_data



# Thinking Theme callback
@app.callback(
    Output("theme_thinking", component_property="children"),
    [
        Input('submit-theme', 'n_clicks'),
        Input("data_storage", "data")
    ]
)
def theme_thinking(clicks, data):

    if clicks == 0:
        return ""

    elif clicks == 1 and "understanding_comments" in pd.DataFrame(data).columns:
        return "Generating response... this may take a few minutes"

# all_text_out results
@app.callback(
    Output("all_text_out", component_property="children"),
    [
        Input("data_storage", "data"),
        Input('submit-theme', 'n_clicks')
    ],
)
def get_overall_text(data, clicks):

    if clicks == 0:
        return '''Theme description will appear here...
                '''
    else:

        data_table = pd.DataFrame.from_dict(data, orient='index').T

        # Prevents error in start up
        if ("understanding_comments" not in data_table.columns):
            return "Please input your data file before running the analysis! - OR you did not label your data column as 'understanding_comments'"

        #data_table['understanding_comments'] = data_table['understanding_comments'].apply(lambda x: x[:100])

        print("****************START*****************")
        out = get_themes3({ 'text': data_table['understanding_comments'].to_list(),
                                     'used_tokens': 0 })

        text = out['text']
        tokens = out['used_tokens']
        summary = out['summary']

        text = text.replace("Themes:", "")
        text = text.replace("Themes", "")


        text = f'''
**Overall Summary**
{summary}


**Themes Extracted**
{text} 
 
 
 
**Total used tokens: { tokens }**

**Approximate cost of analysis: ${ np.round(tokens/1000 * 0.02, decimals=2) }**'''
        

        return text



# Results table
@app.callback(
    Output("topic_table", component_property="children"),
    [
        Input("data_storage", "data"),
        Input('submit-topic', 'n_clicks')
    ],
)
def plot_table(data, clicks):

    if clicks == 0:
        return "Comment topics will appear here..."

    data_table = pd.DataFrame(data)

    # Prevents error in start up
    if ("understanding_comments" not in data_table.columns):
        return "Please input your data file before running the analysis! - OR you did not label your data column as 'understanding_comments'"
    
    data_table = pd.DataFrame(get_topics(data_table))

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
        #style = {"border-style":"solid",
        #         "borderWidth": "2px",
        #         "border-color":"black"},
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
        df["Full-GPT-text-davinci-003"] = ['0', '0', '0']
    
    if ("Full-GPT-text-davinci-003" in df.columns):
        df = df.drop(columns = ['Full-GPT-text-davinci-003'])

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
