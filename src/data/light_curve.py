import glob
import os
from io import StringIO

import numpy as np
import pandas as pd
import joblib


class LightCurve():
    def __init__(self, times, measurements, errors, survey=None, name=None,
                 best_period=None, best_score=None, label=None, p=None,
                 p_signif=None, p_class=None, ss_resid=None):
        self.times = times
        self.measurements = measurements
        self.errors = errors
        self.survey = survey
        self.name = name
        self.best_period = best_period
        self.best_score = best_score
        self.label = label
        self.p = p
        self.p_signif = p_signif
        self.p_class = p_class
        self.ss_resid = ss_resid

    def __repr__(self):
        return "LightCurve(" + ', '.join("{}={}".format(k, v)
                                         for k, v in self.__dict__.items()) + ")"

    def __len__(self):
        return len(self.times)

    def split(self, n_min=0, n_max=np.inf):
        inds = np.arange(len(self.times))
        splits = [np.array(x)
                  for x in np.array_split(inds, np.arange(n_max, len(inds), step=n_max))
                  if len(x) >= n_min]
        return [LightCurve(survey=self.survey, name=self.name,
                           times=self.times[s],
                           measurements=self.measurements[s],
                           errors=self.errors[s], best_period=self.best_period,
                           best_score=self.best_score, label=self.label,
                           p=self.p, p_signif=self.p_signif, p_class=self.p_class,
                           ss_resid=self.ss_resid)
                for s in splits]

    def period_fold(self, p=None):
        if p is None:
            p = self.p
        self.times = self.times % p
        inds = np.argsort(self.times)
        self.times = self.times[inds]
        self.measurements = self.measurements[inds]
        self.errors = self.errors[inds]