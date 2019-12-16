# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:46:25 2019

@author: steph
"""

from primesieve import n_primes
import numpy as np
import copy

class UniHashFam:
    
    def __init__(self, keys, bins, R, prime_pool=None):
        if type(keys)==int: 
            self.keys = list(range(keys))
            self.n_keys = keys
        else: 
            self.keys = keys
            self.n_keys = len(keys)
        self.bins = bins
        self.max_val = max(self.keys)
        self.hash_dicts = []
        self.hash_funcs = []
        self.R = R
        if prime_pool is None:
            self.prime_pool = max([self.bins, self.R])*10
        else: self.prime_pool = prime_pool
    
    def gen_random_params(self, a=None, b=None, p=None):
        if p is None:
            p_list = n_primes(self.prime_pool, self.max_val)
            p = np.random.choice(p_list)
        if a is None:
            a = 2*np.random.randint(1, int((p-1)/2)) + 1
        if b is None:
            b = np.random.randint(0, p-1)
        return (a, b, p)
    
    def gen_hash_family(self, a_list=None, b_list=None, p_list=None, num_dicts=None, pre_hashed_dicts=[], pre_hashed_funcs=[], random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        if num_dicts is None:
            num_dicts = self.R
        new_dicts = []
        new_funcs = []
        for ind in range(num_dicts):
            temp_params = self.gen_random_params(a=self._reroute(a_list, ind), b=self._reroute(b_list, ind), 
                                                     p=self._reroute(p_list, ind))
            temp_dict = self.hash_dict(temp_params)
            temp_func = self.hash_func(temp_params)
            if temp_dict not in pre_hashed_dicts:
                new_dicts.append(temp_dict)
                new_funcs.append(temp_func)
        hash_dicts = pre_hashed_dicts + new_dicts
        hash_funcs = pre_hashed_funcs + new_funcs
        if len(new_dicts)==num_dicts:
            self.hash_dicts = hash_dicts
            self.hash_funcs = hash_funcs
            return self
        else:
            diff = num_dicts - len(new_dicts)
            return self.gen_hash_dicts(a_list, b_list, p_list, diff, hash_dicts, hash_funcs, random_state=np.random.randint(0, 36e5))

    def _reroute(self, some_list, n):
        if some_list is None:
            return None
        elif type(some_list)==int:
            return some_list
        else:
            return some_list[n]
    
    def hash_dict(self, params):
        return {x: self.hash_func(params)(x) for x in self.keys}
    
    def hash_func(self, params):
        return lambda x: int(((params[0] * x + params[1]) % params[2]) % self.bins)
    
    
class MACH:
    
    def __init__(self, model, agg_model, n_cats, R, hash_family):
        self.model = model
        self.R = R
        self.models = []
        self.agg_model = agg_model
        self.n_cats = n_cats
        self.is_fit = False
        if type(hash_family[0])==dict:
            self.hasher = lambda i, n: hash_family[i][n]
        else:
            self.hasher = lambda i, n: hash_family[i](n)
    
    def _hash_list(self, i, y):
        return np.array([self.hasher(i, _) for _ in y])
        
    def _train_subs(self, X, y, **fit_params):
        model_list = []
        for ndx in range(self.R):
            clf = copy.copy(self.model)
            clf = clf.fit(X, self._hash_list(ndx, y), **fit_params)
            model_list.append(clf)
        self.models = model_list
        return self
    
    def _agg_subs(self, X):
        stacked = []
        for model in self.models:
            preds = np.array(list(map(self._int_to_dummies, model.predict(X))))
            stacked.append(preds)
        return np.hstack(stacked)
    
    def _train_meta(self, X, y, **fit_params):
        X = self._agg_subs(X)
        model = copy.copy(self.agg_model)
        model = model.fit(X, y)
        self.agg_model = model
        return self
    
    def fit(self, X, y, sub_fit_params={}, sub_agg_params={}):
        self._train_subs(X, y, **sub_fit_params)
        self._train_meta(X, self._agg_subs(y), **sub_agg_params)
        self.is_fit = True
        return self
    
    def predict(self, X):
        X = self._agg_subs(X)
        return self.agg_model.predict(X)
    
    def predict_proba(self, X):
        X = self._agg_subs(X)
        return self.agg_model.predict_proba(X)
             
    def _int_to_dummies(self, cat):
        vect = np.zeros(self.n_cats, int)
        vect[cat] = 1
        return vect
    
    def _dummies_to_int(self, dummies):
        return np.where(dummies==1)[0][0]
