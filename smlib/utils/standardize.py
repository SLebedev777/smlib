# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""

from sklearn.preprocessing import StandardScaler

class Standardizer(StandardScaler):
    def __init__(self, with_mean=True, with_std=True):
        super().__init__(with_mean=with_mean, with_std=with_std)
