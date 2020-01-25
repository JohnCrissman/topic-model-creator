## reading in medical dataset and manipulating it.
## using this for testing and getting the program working

import csv
import pandas as pd  

# read by default 1st sheet of an excel file
df = pd.read_excel('Tracking_Log_1-10_for_NEIU_excel.xlsx')

print(df.head())
df.to_csv('aa_testing1.csv', encoding='utf-8', index=False)

dfObj1 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj2 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj3 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj4 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj5 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj6 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj7 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj8 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj9 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj10 = pd.DataFrame(columns=['navigation_comments', 'barrier'])

dfObj1['navigation_comments'], dfObj1['barrier'] = df['navigation_comments1'], df['barrier1']
dfObj2['navigation_comments'], dfObj2['barrier'] = df['navigation_comments2'], df['barrier2']
dfObj3['navigation_comments'], dfObj3['barrier'] = df['navigation_comments3'], df['barrier3']
dfObj4['navigation_comments'], dfObj4['barrier'] = df['navigation_comments4'], df['barrier4']
dfObj5['navigation_comments'], dfObj5['barrier'] = df['navigation_comments5'], df['barrier5']
dfObj6['navigation_comments'], dfObj6['barrier'] = df['navigation_comments6'], df['barrier6']
dfObj7['navigation_comments'], dfObj7['barrier'] = df['navigation_comments7'], df['barrier7']
dfObj8['navigation_comments'], dfObj8['barrier'] = df['navigation_comments8'], df['barrier8']
dfObj9['navigation_comments'], dfObj9['barrier'] = df['navigation_comments9'], df['barrier9']
dfObj10['navigation_comments'], dfObj10['barrier'] = df['navigation_comments10'], df['barrier10']

frames = [dfObj1, dfObj2, dfObj3, dfObj4, dfObj5, dfObj6, dfObj7, dfObj8, dfObj9, dfObj10]

result = pd.concat(frames)

# #2
# dfObj['navigation_comments'] = df['navigation_comments2']
# dfObj['barrier'] = df['barrier2']

# dfObj['navigation_comments'] = df['navigation_comments1']
# dfObj['barrier'] = df['barrier1']

# dfObj['navigation_comments'] = df['navigation_comments1']
# dfObj['barrier'] = df['barrier1']

# dfObj['navigation_comments'] = df['navigation_comments1']
# dfObj['barrier'] = df['barrier1']

# dfObj['navigation_comments'] = df['navigation_comments1']
# dfObj['barrier'] = df['barrier1']

# dfObj['navigation_comments'] = df['navigation_comments1']
# dfObj['barrier'] = df['barrier1']

# dfObj['navigation_comments'] = df['navigation_comments1']
# dfObj['barrier'] = df['barrier1']

# dfObj['navigation_comments'] = df['navigation_comments1']
# dfObj['barrier'] = df['barrier1']

# dfObj['navigation_comments'] = df['navigation_comments1']
# dfObj['barrier'] = df['barrier1']
print(result.head())
result.to_csv('aaaaa_testing1.csv', encoding='utf-8', index=False)



''' 
testing pandas dataframe functions 
'''
print(df.iloc[0][0]) # 1
print(df.iloc[4][0]) # 7
print(df.loc[0]['barrier10']) # 5.0

s = df.loc[1]['navigation_comments1']
print(s)
# s = df.loc[3]['navigation_comments1']
if "pt is very" in s:
    print('good to go')
else:
    print('did not work!')
