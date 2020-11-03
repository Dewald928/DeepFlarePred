import matplotlib.pyplot as plt

from sunpy.timeseries import TimeSeries
from sunpy.time import TimeRange, parse_time
from sunpy.net import hek, Fido, attrs as a
import sunpy.map

import drms #querying JSOC data
import astropy.time

"""## Set time range

*Queried items:*
* Class
* Strength
* NOAA AR
* Event date
* Start time
* End time
* Peak Intensity
"""

tr = TimeRange(['2017-09-10 01:00', '2017-09-12 23:00']) #2010/05/01-2018/06/20
results = Fido.search(a.Time(tr), a.Instrument('XRS'))

"""Then download the data and load it into a TimeSeries"""

files = Fido.fetch(results)
goes = TimeSeries(files)

client = hek.HEKClient()
# flares_hek = client.search(hek.attrs.Time(tr.start, tr.end), hek.attrs.EventType('CE'))
flares_hek = client.search(hek.attrs.Time(tr.start, tr.end), hek.attrs.FL, hek.attrs.FRM.Name == 'SWPC')

print([elem["frm_name"] for elem in flares_hek])
print(flares_hek[0])

print(flares_hek.colnames)
print(flares_hek['fl_goescls', 'ar_noaanum', 'event_starttime', 'event_peaktime', 'event_endtime'])

fig, ax = plt.subplots()
goes[1].plot()
ax.axvline(parse_time(flares_hek[4].get('event_peaktime')).plot_date)
ax.axvspan(parse_time(flares_hek[4].get('event_starttime')).plot_date,
           parse_time(flares_hek[4].get('event_endtime')).plot_date,
           alpha=0.2, label=flares_hek[4].get('fl_goescls'))
ax.legend(loc=2)
ax.set_yscale('log')
plt.show()

"""## Get SHARP data
For each records in goes, get SHARP data.

* End time = flare peak intensity
* Start time = based on how many frames needed at 1 hour cadence
"""

c = drms.Client()
print(c.pkeys('hmi.sharp_cea_720s'))
SHARP_search = Fido.search(a.Time(flares_hek[4].get('event_starttime'), flares_hek[4].get('event_peaktime')),
                           a.jsoc.Series('hmi.sharp_cea_720s'),
                           a.jsoc.Notify('dewald123@rocketmail.com'))

print(SHARP_search)

"""Now to download the SHARP data"""

# SHARP_downloaded = Fido.fetch(SHARP_search, path='./Data/SHARP/')

# print(SHARP_downloaded)

SHARP = sunpy.io.read_file('./DATA/hmi.sharp_cea_720s.7117.20170910_153600_TAI.magnetogram.fits')
print(SHARP)


mymap1 = sunpy.map.Map('./DATA/hmi.sharp_cea_720s.7117.20170910_153600_TAI.bitmap.fits')
mymap2 = sunpy.map.Map('./DATA/hmi.sharp_cea_720s.7117.20170910_153600_TAI.Bp.fits')
mymap3 = sunpy.map.Map('./DATA/hmi.sharp_cea_720s.7117.20170910_153600_TAI.Br.fits')
mymap4 = sunpy.map.Map('./DATA/hmi.sharp_cea_720s.7117.20170910_153600_TAI.Bt.fits')
mymap5 = sunpy.map.Map('./DATA/hmi.sharp_cea_720s.7117.20170910_153600_TAI.conf_disambig.fits')
mymap6 = sunpy.map.Map('./DATA/hmi.sharp_cea_720s.7117.20170910_153600_TAI.continuum.fits')
mymap7 = sunpy.map.Map('./DATA/hmi.sharp_cea_720s.7117.20170910_153600_TAI.Dopplergram.fits')
mymap8 = sunpy.map.Map('./DATA/hmi.sharp_cea_720s.7117.20170910_153600_TAI.magnetogram.fits')

mymap1.peek()
mymap2.peek()
mymap3.peek()
mymap4.peek()
mymap5.peek()
mymap6.peek()
mymap7.peek()
mymap8.peek()

"""## Matching NOAA AR number and GOES data

1. NOAA number in HARP = GOES records
2. Location or AR is in +-68 degrees from centrel meridian (projection effects[bobra,2015])
3. Time is before peak intensity time
"""
print('Done...')

