# hmi.sharp_720s[3542][2013.12.28_02:12:00_TAI]

import drms

ids = ['T_REC', 'NOAA_AR', 'HARPNUM', 'CRVAL1', 'CRVAL2', 'CRLN_OBS',
       'CRLT_OBS', 'LAT_FWT', 'LON_FWT']
sharps = ['USFLUX', 'MEANGBT', 'MEANJZH', 'MEANPOT', 'SHRGT45', 'TOTUSJH',
          'MEANGBH', 'MEANALP', 'MEANGAM', 'MEANGBZ', 'MEANJZD', 'TOTUSJZ',
          'SAVNCPP', 'TOTPOT', 'MEANSHR', 'AREA_ACR', 'R_VALUE', 'ABSNJZH']
c = drms.Client()
data_sharp = c.query('hmi.sharp_720s[3542][2013.12.28_02:12:00_TAI]',
                     key=ids+sharps)
