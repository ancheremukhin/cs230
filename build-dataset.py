#!/usr/bin/env python
import csv
from datetime import datetime
import copy


def foreach(*datasets):
    '''
    Open datasets and read row by row skipping header
    '''
    for dataset in datasets:

        if not dataset.endswith('.csv'):
            dataset += '.csv'

        with open('datasets/{0}'.format(dataset), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # skip header
            next(reader)
            for row in reader:
                yield list(map(lambda x: x.strip(), row))


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

# coupon areas
coupon_areas = dict()
for row in foreach('coupon_area_train', 'coupon_area_test'):
    coupon_areas[row[2]] = {
        'small_area': row[0],
        'prefecture': row[1],
        'coupon_id': row[2],
    }

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

# build customer interactions
interactions = list()
for visit in coupon_visits:

    # merge coupon info
    coupon = coupons.get(visit['coupon_id'], None)
    if not coupon:
        print("No coupon {0}".format(visit['coupon_id']))
        continue

    interaction = {}
    interaction['coupon_id'] = coupon['coupon_id']
    interaction['discount_rate'] = coupon['discount_rate']
    interaction['list_price'] = coupon['list_price']
    interaction['discount_price'] = coupon['discount_price']
    interaction['sales_release_date'] = coupon['sales_release_date']
    interaction['sales_end_date'] = coupon['sales_end_date']
    interaction['sales_period'] = coupon['sales_period']

    # merge coupon details
    detail = None
    if visit['purchased']:
        detail = coupon_details.get(visit['purchase_id'], None)
        assert detail['customer_id'] == visit['customer_id']

    interaction['purchased_count'] = detail['purchased_count'] if detail else 0

    # merge customer info
    user = users.get(visit['customer_id'], None)
    assert user

    interaction['age'] = user['age']
    interaction['reg_date'] = user['reg_date']
    interaction['sex'] = user['sex']

    interaction['customer_id'] = visit['customer_id']
    interaction['purchased'] = visit['purchased']
    interaction['view_date'] = visit['view_date']

    interactions.append(interaction)

interactions = sorted(
    interactions,
    key=lambda x: [x['customer_id'], x['view_date']]
)

print(
    "Built {0} interactions with {1} unique customers".format(
        len(interactions),
        len(set(map(lambda x: x['customer_id'], interactions)))
    )
)

with open('user_interactions.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(interactions[0].keys())
    for interaction in interactions:
        csvwriter.writerow(interaction.values())

# build customer trajectories
trajectories = list()
trajectory = {'customer_id': None}

for interaction in interactions:

    if interaction['customer_id'] != trajectory['customer_id']:
        trajectory = {
            # history
            'visit': 0,
            'visit_time_recency': 0,
            'success': 0,
            'success_time_recency': 0,
            'purchased': 0,
            'spent': 0,
            # customer
            'customer_id': interaction['customer_id'],
            'age': interaction['age'],
            'reg_date': interaction['reg_date'],
            'sex': interaction['sex'],
            # view date
            'view_date': interaction['view_date'],
        }

    # state
    trajectory['visit_time_recency'] = (
        interaction['view_date'] - trajectory['view_date']
    ).total_seconds()

    trajectory['success_time_recency'] += (
        interaction['view_date'] - trajectory['view_date']
    ).total_seconds()

    trajectory['view_date'] = interaction['view_date']

    # action
    trajectory['discount_rate'] = interaction['discount_rate']
    trajectory['list_price'] = interaction['list_price']
    trajectory['discount_price'] = interaction['discount_price']
    trajectory['sales_release_date'] = interaction['sales_release_date']
    trajectory['sales_end_date'] = interaction['sales_end_date']
    trajectory['sales_period'] = interaction['sales_period']

    # reward
    trajectory['purchased'] = interaction['purchased']

    # trajectory
    trajectories.append(copy.deepcopy(trajectory))

    trajectory['visit'] += 1
    if trajectory['purchased']:
        trajectory['success_time_recency'] = 0
        trajectory['success'] += 1

        trajectory['purchased'] += interaction['purchased_count']
        trajectory['spent'] += interaction['purchased_count'] * interaction['discount_price']

print("Built {0} trajectories".format(
    len(trajectories))
)

with open('user_trajectories.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(trajectories[0].keys())
    for trajectory in trajectories:
        csvwriter.writerow(trajectory.values())
