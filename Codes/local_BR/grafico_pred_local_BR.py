#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 21:11:48 2021

@author: fellypesb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url_test = 'https://github.com/fellypesb/projeto_PET_2021/raw/main/Dados/local_BR/test_local_BR.csv'


E0 = pd.read_csv(url_test, usecols=['ghi'])
E1 = np.loadtxt('/home/fellypesb/Documents/projeto_PET_2021/Dados/local_BR/pred_E1.csv',delimiter=',')
E2 = np.loadtxt('/home/fellypesb/Documents/projeto_PET_2021/Dados/local_BR/pred_E2.csv',delimiter=',')
E3 = np.loadtxt('/home/fellypesb/Documents/projeto_PET_2021/Dados/local_BR/pred_E3.csv',delimiter=',')
E4 = np.loadtxt('/home/fellypesb/Documents/projeto_PET_2021/Dados/local_BR/pred_E4.csv',delimiter=',')


plt.figure(figsize=(13,5), dpi=300)
#plt.title('Melhor modelo E1', fontsize=12, weight='bold')
plt.xlabel('Tempo (h)', fontsize=12, weight='bold')
plt.ylabel('Irradiância (w/m²)', fontsize=12, weight='bold')
plt.plot(E0,'-', label='DADOS REAIS', linewidth=2)
plt.plot(E1,'--', label='CASO DE ESTUDO E1', linewidth=2)
plt.plot(E2,'-.', label='CASO DE ESTUDO E2', linewidth=2)
plt.plot(E3,':', label='CASO DE ESTUDO E3', linewidth=2)
plt.plot(E4,'--', label='CASO DE ESTUDO E4', linewidth=2)
plt.legend(loc='best', shadow=True,)
plt.xlim(0, 80)
plt.ylim(top=1400)
plt.grid(alpha=0.4)
#plt.savefig('modelo.png', dpi=300)     
