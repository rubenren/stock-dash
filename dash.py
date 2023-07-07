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
        df_raw[df_raw.tic == ticker].to_csv('raw_data/' + ticker + '_' + str(datetime.date.now()) + '.csv') # should be changed t peek into the data


def add_ticker(new_tic):
    tickers = get_list()
    tickers.append(new_tic)
    with open('./tickers.txt', 'w') as f:
        f.write('\n'.join(tickers))


def remove_ticker(tic = ""):
    tickers = get_list()
    tickers.remove(tic)
    with open('./tickers.txt', 'w') as f:
        f.write('\n'.join(tickers))


def get_list():
    if 'tickers.txt' not in listdir('./'):
        return []
    with open('./tickers.txt', 'r') as f:
        out_list = f.read().splitlines()

        return out_list

        
def data_page():
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
    layout = [[sg.Button('Data'), sg.Button('Train')],
              [sg.Button('Eval'), sg.Button('Use')]]

    window = sg.Window('Main Menu', layout, finalize=True)
    return window

    
def eval_page():
    pass


sg.theme('Dark Blue 3')

check_and_make_directories([TRAINED_MODEL_DIR, './raw_data', './clean_data'])


def main():
    mainWin = main_page()
    
    dataWin, trainWin, evalWin, useWin = [None] * 4
    
    check_and_make_list()
    
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
                continue

        if event == 'Load Model':
            continue

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