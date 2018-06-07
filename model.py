#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

geolocator = Nominatim ()

class WineDatabase ( object ):
    def __init__ ( self, myhome='Princeton, NJ 08540', load_locations=False ):
        '''
        Create a wine database from WineReviews cleaned data.

        Parameters
        ----------
        myhome : str, default='Princeton, NJ 08540'
         Address to be used to calculate distances from user
        load_locations : bool, default=True
         if True, will query province locations to generate lat/lon coordinates
         for the wines
        '''
        if myhome is None:
            print ("No home address provided, will not calculate winery distances")

        df = pd.read_csv ('./winemag-data-130k-v2.csv', index_col=0)
        self.df = df
        
        if load_locations:
            self.mylocation = geolocator.geocode(myhome)
            self.locate_provinces ()

        

    def load_model ( self, modelname='./savmodel.pkl', vecname='./savvectorizer.pkl'):
        self.model = joblib.load( modelname )        
        self.tf_vectorizer = joblib.load ( vecname )
        
        
        if 'p_1' not in self.df.columns:
            tf_names = self.tf_vectorizer.transform ( self.df['description'].dropna() )
            pcol = ['p_%i' % i for i in range(10)]
            tpdf = self.model.transform(tf_names)
            for pc in pcol:
                self.df[pc] = 0.
            self.df[pcol] = tpdf
            
    def save_model ( self, fname='./savmodel.pkl', vecname='./savvectorizer.pkl'):
        joblib.dump ( self.model, 'savmodel.pkl')
        joblib.dump ( self.tf_vectorizer, 'savvectorizer.pkl' )

    def construct_model ( self ):
        df = self.df
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=2000,
                                        stop_words='english')

        tf_names = tf_vectorizer.fit_transform ( df['description'].dropna() )
        model = decomposition.LatentDirichletAllocation ( )
        model.fit ( tf_names )

        self.model = model
        self.tf_vectorizer = tf_vectorizer
        #return tf_vectorizer, model

    def locate_provinces ( self ):
        provinces = self.df['province'].unique ()
        print ("Locating %i provinces..." % provinces.size)
        provdf = pd.DataFrame ( index=provinces, columns=['lat','lon','country', 'r_alwin'] )
        for i,cprov in enumerate(provinces):
            location = geolocator.geocode( cprov )
            if location is None:
                location = geolocator.geocode( cprov.split()[-1] )
            if location is None:
                print('Could not parse %s' % cprov)
                continue
            provdf.loc[cprov, 'lat'] = location.latitude
            provdf.loc[cprov, 'lon'] = location.longitude
            provdf.loc[cprov, 'country'] = location.address.split(',')[-1].strip()
            provdf.loc[cprov, 'r_alwin'] = geodesic(location.point, self.mylocation.point).miles

            if i % 100 == 0:
                print ( '...processed %i provinces!' % i )
        self.geowines = provdf

    def load_provinces (self, provinces=None):
        if provinces is None:
            provinces = self.geowines
        hasloc = self.df['province'].dropna()
        provinces = provdf.loc[hasloc, 'r_alwin']
        provinces.index = hasloc.index
        self.df.loc[hasloc.index, 'r_alwin'] = provinces


    def suggest ( self, mydescription,
                  max_price=20.,
                  min_price=0.,
                  min_points=90.,
                  max_distance=1e8, # in miles
                  nsuggest=3,
                  verbose=True ):
        tf_mydescrip = self.tf_vectorizer.transform ( [mydescription] )
        ctopic = self.model.transform( tf_mydescrip )
        if verbose:
            tf_feature_names = self.tf_vectorizer.get_feature_names () 
            for idx, ct in enumerate(ctopic[0]):                
                good_feats = [tf_feature_names[i] for i in self.model.components_[idx].argsort()[:-4 - 1:-1]]
                print('Pr(%i) = %.4f [%s]' % (idx,ct, ', '.join(good_feats)))

        cdf = self.df.query('(points>%i)&(price<%.2f)&(price>%.2f)&(r_alwin<%i)' % (min_points,max_price,min_price,
                                                                                    max_distance))
        pcol = ['p_%i' % i for i in range(10)]
        indices = cdf.index[np.sum((ctopic - cdf[pcol])**2, axis=1).argsort()]

        for i in range(nsuggest):
            hit = self.df.loc[indices[i]]

            print ( '''
%s (%s)
winery: %s
points: %i
price: $%.2f
---
%s''' % ( hit.title, hit.variety, hit.winery, hit.points, hit.price,
                  hit.description ))

    def suggest_at_pricepoints (self, description, pricepoints=[20.,50.,100.], min_points=90.):
        print('User description: %s' % description)
        
        for i,ppoint in enumerate(pricepoints):
            print('''
==========================
        Price point = $%i
==========================''' % ppoint)
            if i == 0:
                min_price = 0.
            else:
                min_price = pricepoints[i-1]
            self.suggest (description, min_price = min_price, max_price=ppoint,
                          min_points=min_points, nsuggest=1, verbose=False )

            
    def getloc ( self, row ):        
        if type(row) is int:
            row = self.df.loc[row]
        string = ', '.join(row[['region_1','region_2','province']].dropna().values)

        location = geolocator.geocode (string)
        tries = row[['region_1','region_2','province']].dropna().tolist()
        tries += [ x.split()[-1] for x in tries]

        i=0
        while location is None:
            string = tries[i]
            location = geolocator.geocode (string)
            i+=1
            if i == len(tries):
                return None
        return location

def viz_wineppfrontier ( df ):
    cdf = df.query('price<100.')
    plt.hist2d ( cdf['price'], cdf.points, bins=[np.arange(4,40),np.arange(80,100)] )
    cfit = optimize.curve_fit ( fn, cdf.price, cdf.points )[0]
    plt.plot ( np.arange(4,40), fn(np.arange(4,40), *cfit), lw=2, color='r')

    yerr = (cdf.points - fn(cdf.price, *cfit)).std()
    for i in range(1,4):
        plt.plot ( np.arange(4,40), fn(np.arange(4,40), *cfit) + yerr*i, lw=2, color='r',
            alpha=i/3., linestyle='--')

    plt.xlim(plt.xlim()[::-1])
    plt.xlabel ( 'price (USD)' )
    plt.ylabel ( 'points' )
    plt.title ( 'Wine points-price tradeoff' )


