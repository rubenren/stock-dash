import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from os import listdir

import itertools


def check_and_make_list():
    '''Creates tickers.txt if it does not exist, otherwise it updates it with respect to stored data'''
    if 'tickers.txt' not in listdir('./'):
        with open('./tickers.txt','w') as file:
            tics = [x.split('_')[0] for x in listdir('./raw_data/')]
            for tic in sorted(tics):
                file.write(tic + '\n')
    else:
        orig_tics = get_list()
        with open('./tickers.txt', 'w') as f:
            new_tics = [x.split('_')[0] for x in listdir('./raw_data/')]
            tics = list(set(orig_tics) | set(new_tics))
            f.write('\n'.join(sorted(tics)))
            
def get_date():
    '''Returns the current date'''
    return datetime.datetime.now().date()

def download_tickers(start_date='2000-01-01', end_date=None):
    
    if not end_date:
        end_date = str(get_date())

    # check the list file
    ticker_list = get_list()
    
    # Gather the tics we have yet to download
    ticker_list = set(ticker_list).difference(set([x.split('_')[0] for x in listdir('./raw_data/')]))
    ticker_list = list(ticker_list)

    # If we have nothing to download, skip
    # TODO: add date checking to keep the database upto date
    if not ticker_list:
        return
    
    # Download the data for the specified date
    df_raw = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=ticker_list).fetch_data()
    
    # Split and store the data with labels
    for ticker in ticker_list:
        df_raw[df_raw.tic == ticker].to_csv('raw_data/' + ticker + '_' + end_date + '.csv') # TODO: should be labelled using their date range instead of update day


def add_ticker(new_tic):
    '''Add a tic marker to the list'''
    tickers = get_list()
    tickers.append(new_tic)
    with open('./tickers.txt', 'w') as f:
        f.write('\n'.join(sorted(tickers)))


def remove_ticker(tic = ""):
    '''Remove a tic marker from the list'''
    tickers = get_list()
    tickers.remove(tic)
    with open('./tickers.txt', 'w') as f:
        f.write('\n'.join(sorted(tickers)))


def get_list():
    '''Read from our list if it exists, otherwise return empty list'''
    if 'tickers.txt' not in listdir('./'):
        return []
    with open('./tickers.txt', 'r') as f:
        out_list = f.read().splitlines()

        return out_list


def load_dataset():
    root_str = './raw_data/'
    files = [x + '_2023-07-13.csv' for x in get_list()]
    total_table = pd.read_csv(root_str + files[0])
    print("Loading files...")
    print(f'{len(files)} files loaded!')
    for filename in files[1:]:
        temp = pd.read_csv(root_str + filename)
        total_table = pd.concat([total_table, temp], axis=0)
    total_table = total_table.sort_values('Unnamed: 0').set_index('Unnamed: 0')
    return total_table


def prep_data(train_start=None, trade_start=None, trade_end=None):
    # Load the entire dataset first
    raw_df = load_dataset()

    # print(raw_df.shape)
    # print(raw_df.describe())
    # print(raw_df.isna().sum())

    min_date = raw_df['date'].min()
    max_date = raw_df['date'].max()
    date_list = list(pd.date_range(min_date, max_date).astype(str))

    if not train_start: train_start = min_date
    if not trade_end: trade_end = max_date
    if not trade_start:
        trade_start = date_list[len(date_list) - len(date_list)//4]

    if f'dataset_{min_date}*{max_date}.csv' in listdir('./clean_data/'):
        processed = pd.read_csv(f'./clean_data/dataset_{min_date}*{max_date}.csv')
    else:
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=True,
            use_turbulence=True,
            user_defined_feature=False
        )

        print(len(raw_df['tic'].unique()), 'tickers before doing the preprocessing')
        raw_df = raw_df.loc[raw_df['date'].between(train_start, trade_end)]

        semi_processed_df = fe.preprocess_data(raw_df)

        # Some processing magic for FinRL indexes
        ticker_list = semi_processed_df['tic'].unique().tolist()
        print(len(semi_processed_df['tic'].unique()), 'tickers after doing the preprocessing')
        combination = list(itertools.product(date_list, ticker_list))

        processed = pd.DataFrame(combination, columns=['date', 'tic']).merge(semi_processed_df, on=['date', 'tic'], how='left')
        processed = processed[processed['date'].isin(semi_processed_df['date'])]
        processed = processed.sort_values(['date', 'tic']).fillna(0)

        # Save the processed data
        processed.to_csv(f'./clean_data/dataset_{min_date}*{max_date}.csv')

    train = data_split(processed, train_start, trade_start)
    test = data_split(processed, trade_start, trade_end)

    return train, test

        
def data_page():
    '''Layout for the data page'''
    col1 = [[sg.T('Start Date:'), sg.InputText(k='-START DATE-')],
            [sg.T('Train End Date:'), sg.InputText(k='-TRAIN END DATE-')],
            [sg.T('Test End Date:'), sg.InputText(k='-TEST END DATE-')]]
    col2 = [[sg.Button('Process'), sg.Button('Close')]]
    layout = [[sg.Text('Database', font='any 16')],
              [sg.Text(k='-TIC NUM-'), sg.Button('List'), sg.Button('Update')],
              [sg.Text('Add or Remove Tickers:'), sg.InputText(k='-TIC-'), sg.Button('Add'), sg.Button('Remove')],
              [sg.HorizontalSeparator()],
              [sg.Text('Live Data', font='any 16')],
              [sg.Column(col1, element_justification='l', expand_x=True),
               sg.Column(col2, element_justification='r', expand_x=True)]]
    
    window = sg.Window('Data Management', layout, finalize=True)
    return window


def main_page():
    '''Layout for the main launcher page'''
    layout = [[sg.Button('Data'), sg.Button('Train')],
              [sg.Button('Eval'), sg.Button('Use')]]

    window = sg.Window('Main Menu', layout, finalize=True)
    return window


def eval_page():
    '''Layout for the evaluation page'''
    models = [x.split(".")[0] for x in listdir(TRAINED_MODEL_DIR)]
    layout = [[sg.Text('Model to load:'), sg.Combo(models ,k='-MODEL FILENAME-')],
              [sg.Button('Load'), sg.Button('Run'), sg.Button('Cancel')],
              [sg.Canvas(size=(640, 480), key='-CANVAS-')]]

    window = sg.Window('Evaluation', layout, finalize=True)

    return window


sg.theme('Dark Blue 3')

check_and_make_directories([TRAINED_MODEL_DIR, './raw_data', './clean_data'])

plt.style.use('ggplot')

# env_kwargs = {
#     'hmax': 100, # Number of shares allowed to buy or sell at any given step
#     'initial_amount': 1_000_000,
#     'num_stock_shares': num_stock_shares,
#     'buy_cost_pct': buy_cost_list,
#     'sell_cost_pct': sell_cost_list,
#     'state_space': state_space,
#     'stock_dim': stock_dimension,
#     'tech_indicator_list': INDICATORS,
#     'action_space': stock_dimension,
#     'reward_scaling': 1e-4
# }

def main():
    mainWin = main_page()
    
    dataWin, trainWin, evalWin, useWin = [None] * 4
    
    check_and_make_list()

    model = None
    train = test = None
    
    while True:
        window, event, values = sg.read_all_windows()
        if window == mainWin and event in ('Exit', sg.WIN_CLOSED):
            break

        # Main page logic
        if window == mainWin:
            if event == 'Data' and not dataWin:
                dataWin = data_page()
            if event == 'Eval' and not evalWin:
                evalWin = eval_page()

        # Data Page Logic
        if window == dataWin:
            if event in ('Close', sg.WIN_CLOSED):
                window.close()
                dataWin = None
            if event == 'Add':
                add_ticker(values['-TIC-'])
            if event == 'Remove':
                remove_ticker(values['-TIC-'])
            if event == 'List':
                tickers = get_list()
                sg.popup_scrolled("\n".join(tickers), title='Tickers', non_blocking=True)
            if event == 'Update':
                download_tickers()
            if event == 'Process':
                train, test = prep_data('2010-01-01')


        # Evaluation Page Logic
        if window == evalWin:
            if event in ('Cancel', sg.WIN_CLOSED):
                window.close()
                evalWin = None
            if event == 'Load':
                # Very temporary

                # Make sure we have data to process first
                if train is None or test is None:
                    print("Cannot load environment without a dataset Processed")
                    continue

                # Define our environment
                stock_dimension = len(train.tic.unique())
                state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
                print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
                buy_cost_list = sell_cost_list = [0.001] * stock_dimension # cost of purchasing
                num_stock_shares = [0] * stock_dimension # Initial stock allocation with only cash for now, will change later

                env_kwargs = {
                    'hmax': 100,
                    'initial_amount': 1_000_000,
                    'num_stock_shares': num_stock_shares,
                    'buy_cost_pct': buy_cost_list,
                    'sell_cost_pct': sell_cost_list,
                    'state_space': state_space,
                    'stock_dim': stock_dimension,
                    'tech_indicator_list': INDICATORS,
                    'action_space': stock_dimension,
                    'reward_scaling': 1e-4
                }

                # Define our gym (testing environment)
                e_test_gym = StockTradingEnv(
                    df = test,
                    turbulence_threshold=70,
                    risk_indicator_col='vix',
                    **env_kwargs
                )
                
                # import our model
                path_to_model = TRAINED_MODEL_DIR + '/' + values['-MODEL FILENAME-']
                print(path_to_model)
                model = TD3.load(path_to_model)
                
                # Now run our predictions and get our account value
                account_value, account_actions = DRLAgent.DRL_prediction(model=model, environment=e_test_gym)
                
                print(account_value)
                print(account_actions)



    window.close()
    if dataWin:
        dataWin.close()
        dataWin = None
    if trainWin:
        trainWin.close()
        trainWin = None
    if evalWin:
        evalWin.close()
        evalWin = None
    if useWin:
        useWin.close()
        evalWin = None

main()