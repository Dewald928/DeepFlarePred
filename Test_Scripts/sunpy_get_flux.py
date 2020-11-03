import matplotlib.pyplot as plt

from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net import hek
from sunpy.time import TimeRange, parse_time
from sunpy.timeseries import TimeSeries
import matplotlib.dates as mdates
import os

# Let’s first grab GOES XRS data for a particular time of interest
tr = TimeRange(['2017-09-06 00:00', '2017-09-06 23:00'])
results = Fido.search(a.Time(tr), a.Instrument.xrs)

# Then download the data and load it into a TimeSeries
files = Fido.fetch(results)
goes = TimeSeries(files)

# Next lets grab the HEK flare data for this time from the NOAA Space Weather Prediction Center (SWPC)
# client = hek.HEKClient()
# flares_hek = client.search(hek.attrs.Time(tr.start, tr.end),
#                            hek.attrs.FL, hek.attrs.FRM.Name == 'SWPC')

fig, ax = plt.subplots(figsize=(10,6))
goes.plot()

# ax.axvline(parse_time(flares_hek[0].get('event_peaktime')).plot_date)
# ax.axvspan(parse_time(flares_hek[0].get('event_starttime')).plot_date,
#            parse_time(flares_hek[0].get('event_endtime')).plot_date,
#            alpha=0.2, label=flares_hek[0].get('fl_goescls'))
# ax.legend(loc=2)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%b/%Y, %H:%M"))
# plt.gca().xaxis.set_major_locator(mdates.HourLocator())
plt.gcf().autofmt_xdate()
ax.set_ylabel('Goes X-Ray Flux \n (Watts $M^{-2}$')
ax.legend(['0.5 - 4.0 Å', '1.0 - 8.0 Å'])
ax.set_yscale('log')
pathname = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/spaceweather/')
plt.tight_layout()
plt.savefig(pathname + 'x-rayflux.png')
plt.show()