import PySimpleGUI as sg
import pandas as pd
import numpy as np
import datetime

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl import config_tickers

from finrl.config import TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories

from os import listdir


def check_and_make_list():
    if 'tickers.txt' not in listdir('./'):
        with open('./tickers.txt','w') as file:
            tics = [x.split('_')[0] for x in listdir('./raw_data/')]
            for tic in tics:
                file.write(tic + '\n')
    else:
        with open('./tickers.txt', 'w') as f:
            orig_tics = get_list()
            new_tics = [x.split('_')[0] for x in listdir('./raw_data/')]
            tics = list(set(orig_tics) | set(new_tics))
            for tic in tics:
                f.write(tic)
            

def download_tickers(start_date='2000-01-01', end_date='2023-06-30'):
    # check the list file
    ticker_list = get_list()
    ticker_list = set(ticker_list).difference(set([x.split('_')[0] for x in listdir('./raw_data/')]))
    ticker_list = list(ticker_list)
    if not ticker_list:
        return
    df_raw = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=ticker_list).fetch_data()
    for ticker in ticker_list:
        df_raw[df_raw.tic == ticker].to_csv('raw_data/' + ticker + '_' + str(datetime.datetime.now()) + '.csv') # should be changed t peek into the data


def add_ticker(new_tic):
    with open('./tickers.txt', 'a') as f:
        f.write(new_tic + '\n')


def get_list():
    if 'tickers.txt' not in listdir('./'):
        return []
    with open('./tickers.txt', 'r') as f:
        out_list = f.read().splitlines()

        return out_list


sg.theme('Dark Blue 3')

check_and_make_directories([TRAINED_MODEL_DIR, './raw_data', './clean_data'])

tickers = get_list()

layout = [[sg.Text('Number of Tickers:'), sg.Text(text=len(tickers) ,size=(4,1), key='-TICNUM-'), sg.Button('List'), sg.Button('Update')],
          [sg.Text('Add or Remove Tickers:'), sg.InputText(), sg.Button('Add'), sg.Button('Remove')],
          [sg.Button('Load Model'), sg.InputText(k='-MODEL-'), sg.Text(key='-LOADED MODEL-')],
          [sg.Button('Ok'), sg.Button('Cancel')]]

window = sg.Window('Window Title', layout)

while True:
    tickers = get_list()
    event, values = window.read()
    window['-TICNUM-'].update(len(tickers))
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    print('You entered ', values[0])
    if event == 'List':
        sg.popup_scrolled("\n".join(tickers), title='Tickers', non_blocking=True)
    if event == 'Update':
        download_tickers()
    if event == 'Add':
        add_ticker(values[0])
    if event == 'Load Model':
        continue

window.close()
