#!/usr/bin/env python
import csv
from datetime import datetime


def foreach(*datasets):
    '''
    Open datasets and read row by row skipping header
    '''
    for dataset in datasets:

        if not dataset.endswith('.csv'):
            dataset += '.csv'

        with open('datasets/{0}'.format(dataset), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # skip header
            reader.next()
            for row in reader:
                yield map(lambda x: x.strip(), row)


# user info
users = dict()
for row in foreach('user_list'):
    users[row[5]] = {
        'customer_id': row[5],
        'age': int(row[2]),
        'reg_date': datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'),
        'sex': True if row[1] == 'm' else False,
    }

# coupon info
# TODO: categorize area: large area, small area, prefecture
coupons = dict()
for row in foreach('coupon_list_train', 'coupon_list_test'):
    coupons[row[23]] = {
        'coupon_id': row[23],
        'discount_rate': int(row[2]),
        'list_price': int(row[3]),
        'discount_price': int(row[4]),
        'sales_release_date': datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S'),
        'sales_end_date': datetime.strptime(row[6], '%Y-%m-%d %H:%M:%S'),
        'sales_period': int(row[7]),

    }
    #    'on_mon': bool(int(row[11])),
    #    'on_tue': bool(int(row[12])),
    #    'on_wed': bool(int(row[13])),
    #    'on_thu': bool(int(row[14])),
    #    'on_fri': bool(int(row[15])),
    #    'on_sat': bool(int(row[16])),
    #    'on_sun': bool(int(row[17])),
    #    'on_holiday': bool(int(row[18])),
    #    'before_holiday': bool(int(row[19])),
    #    'valid_from_date': datetime.strptime(row[8], '%Y-%m-%d'),
    #    'valid_end_date': datetime.strptime(row[9], '%Y-%m-%d'),
    #    'valid_end_date': datetime.strptime(row[10], '%Y-%m-%d'),

# coupon visits
# TODO: categorize referer
coupon_visits = list()
for row in foreach('coupon_visit_train'):
    coupon_visits.append({
        'purchased': bool(int(row[0])),
        'view_date': datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S'),
        'coupon_id': row[4],
        'customer_id': row[5],
    })

    if coupon_visits[-1]['purchased']:
        assert row[7]
        coupon_visits[-1]['purchase_id'] = row[7]

# coupon details
coupon_details = dict()
for row in foreach('coupon_detail_train'):
    coupon_details[row[3]] = {
        'purchased_count': int(row[0]),
        'purchase_date': datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S'),
        'purchase_id': row[3],
        'customer_id': row[4],
        'coupon_id': row[5],
    }

# TODO: coupon_area_train.csv

# denormolize customer trajectories
trajectories = list()
for visit in coupon_visits:

    # merge coupon info
    coupon = coupons.get(visit['coupon_id'], None)
    if not coupon:
        print("No coupon {0}".format(visit['coupon_id']))
        continue

    trajectory = {}
    trajectory['coupon_id'] = coupon['coupon_id']
    trajectory['discount_rate'] = coupon['discount_rate']
    trajectory['list_price'] = coupon['list_price']
    trajectory['discount_price'] = coupon['discount_price']
    trajectory['sales_release_date'] = coupon['sales_release_date']
    trajectory['sales_end_date'] = coupon['sales_end_date']
    trajectory['sales_period'] = coupon['sales_period']

    # merge coupon details
    detail = None
    if visit['purchased']:
        detail = coupon_details.get(visit['purchase_id'], None)
        assert detail['customer_id'] == visit['customer_id']

    trajectory['purchased_count'] = detail['purchased_count'] if detail else 0

    # merge customer info
    user = users.get(visit['customer_id'], None)
    assert user

    trajectory['age'] = user['age']
    trajectory['reg_date'] = user['reg_date']
    trajectory['sex'] = user['sex']

    trajectory['customer_id'] = visit['customer_id']
    trajectory['purchased'] = visit['purchased']
    trajectory['view_date'] = visit['view_date']

    trajectories.append(trajectory)

trajectories = sorted(
    trajectories,
    key=lambda x: [x['customer_id'], x['view_date']]
)

print("Built {0} customer trajectories".format(len(trajectories)))

with open('user_trajectories.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(trajectories[0].keys())
    for trajectory in trajectories:
        csvwriter.writerow(trajectory.values())
