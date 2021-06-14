import pickle


country_map = {'SVN': 39, 'JPN': 14, 'USA': 29, 'IDN': 36, 'IND': 35, 'BRA': 30, 'BEL': 2, 'CAN': 3, 'ROU': 54, 'CZE': 4, 'HRV': 50, 'CHE': 26, 'SVK': 23, 'NOR': 20, 'SAU': 45, 'CYP': 51, 'MDG': 56, 'KOR': 15, 'ESP': 24, 'LVA': 43, 'GBR': 28, 'ZAF': 40, 'DEW': 41, 'FIN': 6, 'FRA': 7, 'POL': 21, 'MLT': 52, 'RUS': 38, 'MEX': 17, 'ZMB': 58, 'ISR': 37, 'DNK': 5, 'PER': 53, 'EU28': 42, 'BGR': 49, 'MAR': 57, 'NLD': 18, 'DEU': 8, 'SWE': 25, 'ARG': 47, 'IRL': 12, 'CRI': 48, 'ISL': 11, 'PRT': 22, 'TUR': 27, 'EST': 34, 'CHN': 32, 'GRC': 9, 'NZL': 19, 'EA19': 46, 'SRB': 59, 'LTU': 44, 'CHL': 31, 'AUT': 1, 'MKD': 55, 'HKG': 60, 'COL': 33, 'ITA': 13, 'LUX': 16, 'HUN': 10, 'AUS': 0}

filename = "decision.sav"
print("Enter country code")
country = country_map[input()]
print("Enter Year")
year = input()
model = pickle.load(open(filename, 'rb'))
print(model.predict([[country,year]])[0])
