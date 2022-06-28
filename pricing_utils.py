import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import date
import os
import shutil
from selfee_lib.utils.file_reader import read_M021
from selfee_lib.utils.database_functions import update_data
from termcolor import colored


def save_template_m21(filename, year, list_prm, list_commune, list_nom=None):
    template_m21 = pd.DataFrame()
    template_m21['PRM'] = list_prm
    try:
        list_nom == None
        template_m21['Nom'] = np.nan
    except ValueError:
        template_m21['Nom'] = list_nom
    template_m21['Prenom'] = np.nan
    template_m21['Denomination sociale'] = list_commune

    template_m21 = template_m21.dropna(how='all')

    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    template_m21['Date debut'] = start_date.strftime("%d/%m/%Y")
    template_m21['Date fin'] = end_date.strftime("%d/%m/%Y")
    template_m21.to_csv(filename, sep=';', index=False)
    print('saved to ', filename)


def load_m021(m21_folder, engine, to_db=True):
    if '.zip' in m21_folder:
        shutil.unpack_archive(m21_folder, m21_folder.split('.zip')[0])
        m21_folder = m21_folder.split('.zip')[0]
    df = pd.DataFrame()
    for file in tqdm(os.listdir(m21_folder)):
        if not 'Rejets' in file:
            df = pd.concat([df, read_M021(m21_folder + '/' + file)])
    if to_db:
        update_data(df, "M021", engine)
    return df


def load_conso(info):
    conso = info.conso
    if len(conso) == 0:
        df = info.BPU
        year = info.info.loc[0, 'valeur']
        list_prm = df.PRM.values.astype(str)

        sql = '''SELECT from_utc_datetime, from_local_datetime, sum(valeur_point_mw) as conso 
        FROM "M021" where prm in ''' + str(tuple(list_prm)) + 'group by from_utc_datetime, from_local_datetime'
        conso = pd.read_sql(sql, info.engine2)

        if len(conso) == 0:
            print('Conso not in database. \n > Loading M021 file....')
            path_M021 = info.info.loc[1, 'valeur']
            if not os.path.exists(path_M021):
                print(' > Creating template for request...')
                path_template = info.info.loc[2, 'valeur']
                list_commune = df.commune.values
                save_template_m21(path_template, year, list_prm, list_commune)
                print(colored(
                    '\nExit : Once M021 files requested, update the path in pricing_template.xlsx and rerun the script',
                    'red'))
                return 0

            conso = load_m021(path_M021, info.engine2)
    return conso.dropna()


def generate_spot_ref(year, month_avg, month_peak_avg, info):
    sql = '''SELECT * FROM "Prices" where date_part('year', from_local_datetime) = ''' + str(
        year) + ''' and type = 'day_ahead' '''
    spot = pd.read_sql(sql, info.engine)
    non_working_days = (spot.from_local_datetime.dt.weekday >= 5) | (
        spot.from_local_datetime.dt.strftime("%Y-%m-%d").isin(info.feries)) | (
                           spot.from_local_datetime.dt.strftime("%Y-%m-%d").isin(info.ponts))
    night_hour = (spot.from_local_datetime.dt.hour < 8) | (spot.from_local_datetime.dt.hour > 20)

    spot['peak'] = True
    spot.loc[non_working_days | night_hour, 'peak'] = False
    spot['month'] = spot.from_local_datetime.dt.month

    for month in range(1, 13):
        new_avg = month_avg[month - 1]
        new_avg_peak = month_peak_avg[month - 1]

        filter_peak = (spot.month == month) & (spot.peak)
        filter_base = (spot.month == month) & (spot.peak == False)
        avg_peak = spot.loc[filter_peak, 'price_euro'].mean()
        avg_base = spot.loc[filter_base, 'price_euro'].mean()
        n_peak = len(spot[filter_peak])
        n_base = len(spot[filter_base])

        new_avg_base = new_avg + (new_avg - new_avg_peak) * n_peak / n_base
        spot.loc[filter_peak, 'prix_spot'] = new_avg_peak / avg_peak * spot.loc[filter_peak, 'price_euro'].values
        spot.loc[filter_base, 'prix_spot'] = new_avg_base / avg_base * spot.loc[filter_base, 'price_euro'].values
    return spot[['from_utc_datetime', 'from_local_datetime', 'prix_spot']]


def couverture_optimale(df):
    df['achats'] = df.conso * df.prix_spot
    print('Couverture optimale =', round(df.achats.sum() / df.prix_spot.sum(), 2), 'MW')


def couts_des_ecarts(df):
    df['weekday'] = df.from_local_datetime.dt.weekday
    df['hour'] = df.from_local_datetime.dt.hour
    mean_conso = df.groupby(['weekday', 'hour']).mean()[['conso']]
    mean_conso.columns = ['prev_conso']
    ecarts = df.merge(mean_conso, left_on=['weekday', 'hour'], right_index=True)

    imbalance_prices = (np.abs(ecarts.prev_conso - ecarts.conso) * 0.1 * ecarts.prix_spot).sum()
    imbalance_cost = np.round(imbalance_prices / ecarts.conso.sum(), 2)
    print('Coûts des écarts =', imbalance_cost, '€/MWh')


def droit_arenh(data, info):
    df = data.copy()
    df['month'] = df.from_local_datetime.dt.month
    df['ferie'] = df.from_local_datetime.dt.strftime("%Y-%m-%d").isin(info.feries)
    df['weekday'] = df.from_local_datetime.dt.weekday
    df['hour'] = df.from_local_datetime.dt.hour
    mask1 = (df.month == 8) | (df.month == 7)
    mask2 = (((df.month >= 4) & (df.month <= 6)) | (df.month == 9) | (df.month == 10))
    mask3 = ((df.weekday == 6) | (df.weekday == 7) | df.ferie | ((df.hour >= 1) & (df.hour < 7)))
    mask4 = mask2 & mask3

    mask = (mask1 | mask4)
    pm = np.mean(df.loc[mask, 'conso'])
    print("ARENH : ", (round(0.964 * pm, 2)), "MW")


def calculate_price(year, month_avg, month_peak_avg, data, info):
    spot = generate_spot_ref(year, month_avg, month_peak_avg, info)
    df = data.merge(spot, on='from_utc_datetime')
    volume_arenh = info.info.loc[4, 'valeur']
    volume_couverture = info.info.loc[7, 'valeur']
    prix_arenh = info.info.loc[5, 'valeur']
    prix_couverture = info.info.loc[8, 'valeur']
    prod_coef = info.info.loc[10, 'valeur']/100
    prix_prod = info.info.loc[11, 'valeur']
    couts_ecarts = info.info.loc[13, 'valeur']

    achats_prod = prod_coef * df['prod'] * (prix_prod - df.prix_spot)
    achat_couverture = volume_couverture * (prix_couverture - df.prix_spot)
    achat_couv_sans_arenh =  (volume_couverture + volume_arenh) * (prix_couverture - df.prix_spot)
    achat_arenh = volume_arenh * (prix_arenh - df.prix_spot)
    df['achats'] = df.conso * df.prix_spot + achat_arenh + achat_couverture + achats_prod
    df['achats_sans_arenh'] = df.conso * df.prix_spot + achat_couv_sans_arenh + achats_prod
    price = np.round(df.achats.sum() / df.conso.sum() + couts_ecarts, 2)
    price_sans_arenh = np.round(df.achats_sans_arenh.sum() / df.conso.sum() + couts_ecarts, 2)
    mean_spot = np.round(df.prix_spot.mean(), 2)
    mean_spot_avg = np.round(month_peak_avg.mean(), 2)
    return mean_spot, mean_spot_avg, price, price_sans_arenh
