import os
import pandas as p
def get_all_spreadsheets(datadir):
    """
    datadir: directory containing the subdirectories 'Codes' and 'Videos'
    return: dict mapping spreadsheet filename to dataframe
    """
    skip = ['TBOP Coding_Stop and Go.xlsx'] # Different format from the other spreadsheets
    spreadsheet_dict = {}
    for root, dirs, files in os.walk(os.path.join(datadir, 'Codes'), topdown=True):
        for file in files:
            if file in skip:
                continue
            filename = os.path.join(datadir, 'Codes', file)
            df = p.read_excel(filename,index_col=None,header=0)
            spreadsheet_dict[file] = df
    return spreadsheet_dict

