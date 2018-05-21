#!/usr/bin/env python
import csv
from datetime import datetime


def foreach(dataset):
    '''
    Open dataset and read row by row skipping header
    '''
    if not dataset.endswith('.csv'):
        dataset += '.csv'

    with open('datasets/{0}'.format(dataset), 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # skip header
        reader.next()
        for row in reader:
            yield row


# customers info
users = list()
for row in foreach('user_list'):
    users.append({
        'id': row[5],
        'age': int(row[2]),
        'reg_date': datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'),
        'sex': True if row[1] == 'm' else False,
    })

# coupon info
coupons = list()
for row in foreach('coupon_list_train'):
    coupons.append({
        'id': row[23],
        'discount_rate': int(row[2]),
        'list_price': int(row[3]),
        'discount_price': int(row[4]),
        'sales_release_date': datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S'),
        'sales_end_date': datetime.strptime(row[6], '%Y-%m-%d %H:%M:%S'),
        'sales_period': int(row[7]),
        'on_mon': bool(row[11]),
        'on_tue': bool(row[12]),
        'on_wed': bool(row[13]),
        'on_thu': bool(row[14]),
        'on_fri': bool(row[15]),
        'on_sat': bool(row[16]),
        'on_sun': bool(row[17]),
        'on_holiday': bool(row[18]),
        'before_holiday': bool(row[19]),

    })
    #'valid_from_date': datetime.strptime(row[8], '%Y-%m-%d'),
    #'valid_end_date': datetime.strptime(row[9], '%Y-%m-%d'),
    #'valid_end_date': datetime.strptime(row[10], '%Y-%m-%d'),

# coupon details
coupon_details = list()
for row in foreach('coupon_detail_train'):
    coupon_details.append({
        'purchased_count': int(row[0]),
        'purchase_date': datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S'),
        'customer_id': row[4],
        'coupon_id': row[5],
    })

# coupon visits
coupon_visits = list()
for row in foreach('coupon_visit_train'):
    coupon_visits.append({
        'purchased': bool(row[0]),
        'view_date': datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S'),
        'coupon_id': row[5],
        'customer_id': row[6],
    })
