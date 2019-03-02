#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:08:28 2019

@author: erkanzileli
"""

import pandas as pd
from geopy import distance
import numpy as np

# iteration count
for i in range(10):

    sellers = [
        {'name':'red', 'distance': distance.distance((0,0), (41.049792, 29.003031)), 'index': 0, 'limit': 20, 'current': 0},
        {'name':'green', 'distance': distance.distance((0,0), (41.069940, 29.019250)), 'index': 1, 'limit': 50, 'current': 0},
        {'name':'blue', 'distance': distance.distance((0,0), (41.049997, 29.026108)), 'index': 2, 'limit': 80, 'current': 0}
        ]

    orders = pd.read_csv('orders.csv')
    orders['seller'] = -1
    orders = orders.iloc[:,:].values
    np.random.shuffle(orders)
    
    total_distance = 0.0

    for order in orders:
        # Distance between order location and origin point (0,0)
        order_distance = distance.distance((0,0), (order[1], order[2]))
    
        # if seller reaches the limit, remove from seller options
        available_sellers = list(filter(lambda s: s['limit'] > s['current'], sellers))
    
        # Initializing the nearest seller to order with first seller
        nearest_seller = {'seller': available_sellers[0], 'distance': abs(available_sellers[0]['distance'] - order_distance), 'index': available_sellers[0]['index']}
    
        for seller in available_sellers:
            if abs(order_distance - seller['distance']) < nearest_seller['distance']:
                nearest_seller['seller'] = seller
                nearest_seller['distance'] = abs(order_distance - seller['distance'])
                nearest_seller['index'] = seller['index']
    
        # we are found the nearest seller, set the order's seller
        order[3] = nearest_seller['index']
        
        # seller sells
        sellers[nearest_seller['index']]['current'] += 1
        
        # add to total_distance the distance between order and seller
        # total_distance += 
        total_distance += nearest_seller['distance'].km
        
    # kmeans_data = np.column_stack((orders[:, 0], orders[:, -1]))
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=3, init="k-means++")
    # kmeans.fit(kmeans_data)
    
    

    print(total_distance)