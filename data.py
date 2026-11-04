
build_station_gt.py
100%
#!/usr/bin/env python3
"""Build ground-truth station rainfall for station_level_interpolater.py.

Inputs : rainfall_rf40_layout.nc (IMD station-month daily RF), NOAA ISD station list (WMO id -> lat/lon).
Outputs: out/station_point_rainfall_{year}.nc  (rainfall[time, station], lat/lon[station]; time = days since year-01-01)
         out/rf40_layout_filled.csv            (rf40_station.csv column format + StationId, lat/lon filled)
         out/station_metadata.csv, out/qc_extreme_values.csv
Rules  : coordinates only via exact WMO-id match (stations without a match are excluded);
         source rainfall values are never altered or imputed; missing stays NaN.
"""
import calendar, re, difflib, sys
from pathlib import Path
import numpy as np, pandas as pd, netCDF4 as nc

SRC = Path(sys.argv[1])      # dir with rainfall_rf40_layout.nc and NDCQ-*/ tables
ISD = Path(sys.argv[2])      # isd-history.csv
OSCAR = [Path(p) for p in sys.argv[4:]]  # OSCAR/Surface station-search JSON pages (fallback source)
OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
EXTREME_MM = 500.0           # reported only, never removed or clamped

# ---- station names from NDCQ day-summary tables
names = {}
for f in sorted(SRC.glob('NDCQ-*/SURFACE_TABLE_2_DAILY_*.csv')):
    cur = None
    for line in open(f):
        if line.startswith('STATION :'): cur = line.split(':', 1)[1].strip()
        elif cur and line[:1].isdigit(): names[line.split(',')[0]] = cur; cur = None

# ---- coordinates from ISD by WMO id (first 5 digits of USAF)
isd = pd.read_csv(ISD, dtype=str)
isd = isd[isd.CTRY == 'IN'].copy()
isd['id'] = isd.USAF.str[:5]
isd['LAT'] = pd.to_numeric(isd.LAT, errors='coerce'); isd['LON'] = pd.to_numeric(isd.LON, errors='coerce')
isd = isd[isd.LAT.notna() & ~((isd.LAT == 0) & (isd.LON == 0))]
norm = lambda s: re.sub(r'[^A-Z]', '', str(s).upper())

d = nc.Dataset(SRC / 'rainfall_rf40_layout.nc')
sid = np.array([b''.join(x).decode() for x in nc.chartostring(d['StationId'][:])]) if False else nc.chartostring(d['StationId'][:])
sid = np.array([s.strip() for s in sid])
year = d['Year'][:].data; month = d['Month'][:].data
rain = np.ma.filled(d['rainfall'][:], np.nan).astype('float64')
status = d['observation_status'][:].data
rain[status != 1] = np.nan   # only 'available_source_value' counts as an observation

meta = []
for i in sorted(set(sid)):
    c = isd[isd.id == i]
    nm = names.get(i, '')
    if c.empty: continue
    sim = lambda r: difflib.SequenceMatcher(None, norm(nm), norm(r.Index_name)).ratio()
    c = c.assign(Index_name=c['STATION NAME'])
    c = c.assign(s=[sim(r) for r in c.itertuples()]).sort_values(['s', 'END'], ascending=False)
    b = c.iloc[0]
    spread = max(c.LAT.max() - c.LAT.min(), c.LON.max() - c.LON.min())
    meta.append(dict(StationId=i, Station=nm or b['STATION NAME'], ndcq_name=nm, isd_name=b['STATION NAME'],
                     isd_usaf=b.USAF, Latitude=b.LAT, Longitude=b.LON, name_similarity=round(b.s, 2),
                     n_isd_candidates=len(c), candidate_spread_deg=round(spread, 3),
                     coord_source='NOAA ISD history (WMO id match)'))
# fallback: WMO OSCAR/Surface catalogue, exact WMO-id match only
import json
have = {m['StationId'] for m in meta}
osc = []
for f in OSCAR: osc += json.load(open(f))['stationSearchResults']
for o in osc:
    loc = str(o.get('wigosId', '')).split('-')[-1]
    if 'India' in str(o.get('territory')) and loc in set(sid) and loc not in have and o.get('latitude') is not None:
        nm = names.get(loc, '')
        meta.append(dict(StationId=loc, Station=nm or o['name'], ndcq_name=nm, isd_name=o['name'], isd_usaf='',
                         Latitude=o['latitude'], Longitude=o['longitude'], name_similarity=round(difflib.SequenceMatcher(None, norm(nm), norm(o['name'])).ratio(), 2) if nm else np.nan,
                         n_isd_candidates=1, candidate_spread_deg=0.0, coord_source='WMO OSCAR/Surface (WMO id match)'))
        have.add(loc)
meta = pd.DataFrame(meta).sort_values('StationId').reset_index(drop=True)
meta.to_csv(OUT / 'station_metadata.csv', index=False)
order = {s: k for k, s in enumerate(meta.StationId)}
print(f'{len(order)} stations with verified-by-id coordinates; {len(set(sid)) - len(order)} excluded')

# ---- NetCDF writers
def write_nc(path, tdays, units, ids_idx, arr, title):
    ds = nc.Dataset(path, 'w'); n = len(ids_idx)
    ds.createDimension('time', arr.shape[0]); ds.createDimension('station', n)
    v = ds.createVariable('time', 'f4', ('time',)); v[:] = tdays; v.units = units
    v = ds.createVariable('station', 'i4', ('station',)); v[:] = np.arange(n)
    v = ds.createVariable('lat', 'f4', ('station',)); v[:] = meta.Latitude.values[ids_idx]; v.units = 'degrees_north'
    v = ds.createVariable('lon', 'f4', ('station',)); v[:] = meta.Longitude.values[ids_idx]; v.units = 'degrees_east'
    v = ds.createVariable('station_id', 'i4', ('station',)); v[:] = meta.StationId.astype(int).values[ids_idx]
    v.long_name = 'WMO station id (IMD INDEX)'
    v = ds.createVariable('rainfall', 'f4', ('time', 'station'), zlib=True, complevel=4, fill_value=np.float32(np.nan))
    v[:] = arr; v.units = 'mm/day'; v.long_name = 'IMD observed daily rainfall (NaN = no valid observation)'
    ds.title = title
    ds.source = 'IMD daily surface table II (RF) via rainfall_rf40_layout.nc; coordinates by WMO id from NOAA ISD / WMO OSCAR'
    ds.comment = 'No imputation, gap filling, clamping or exclusion of values. Stations listed are those with >=1 valid observation in the period.'
    ds.close()

def fill(y, arr, t0):
    seen = set()
    for r in np.where((year == y) & np.isin(sid, list(order)))[0]:
        key = (sid[r], month[r]); assert key not in seen, key; seen.add(key)
        m = int(month[r]); nd = calendar.monthrange(y, m)[1]
        off = t0 + sum(calendar.monthrange(y, k)[1] for k in range(1, m))
        arr[off:off + nd, order[sid[r]]] = rain[r, :nd]

ydays = lambda y: 366 if calendar.isleap(y) else 365
all_years = sorted(set(int(y) for y in year))
ext = []
for y in all_years:
    arr = np.full((ydays(y), len(order)), np.nan, dtype='float32'); fill(y, arr, 0)
    keep = np.where(np.isfinite(arr).any(axis=0))[0]
    for t, k in zip(*np.where(arr > EXTREME_MM)): ext.append((y, meta.StationId[k], int(t), float(arr[t, k])))
    write_nc(OUT / f'station_point_rainfall_{y}.nc', np.arange(ydays(y)), f'days since {y}-01-01', keep, arr[:, keep], f'Station ground-truth daily rainfall {y}')
    print(y, 'stations', len(keep), 'obs fraction %.3f' % np.isfinite(arr[:, keep]).mean())

# combined files for a period (time = days since first Jan 1; station = union of stations with data)
for a, b in [(2018, 2025), (1975, 1990)]:
    ys = [y for y in all_years if a <= y <= b]
    offs = np.cumsum([0] + [ydays(y) for y in ys]); arr = np.full((offs[-1], len(order)), np.nan, dtype='float32')
    for y, o in zip(ys, offs): fill(y, arr, o)
    keep = np.where(np.isfinite(arr).any(axis=0))[0]
    write_nc(OUT / f'station_point_rainfall_{a}_{b}.nc', np.arange(offs[-1]), f'days since {a}-01-01', keep, arr[:, keep], f'Station ground-truth daily rainfall {a}-{b}')
    print(f'{a}-{b}', 'stations', len(keep), 'days', offs[-1])
pd.DataFrame(ext, columns=['year', 'StationId', 'day_of_year_index', 'rainfall_mm']).to_csv(OUT / 'qc_extreme_values_INFO_ONLY.csv', index=False)

# ---- rf40-format CSV, all years (4-digit years; blanks kept blank)
cols = ['StationId', 'Latitude', 'Longitude', 'Year', 'Month', 'Record Type(1 or 2)', 'Number of Days', 'Mean_Row'] + \
       [str(k) for k in range(1, 32)] + ['District', 'Station', 'Classification']
sel = np.where(np.isin(sid, list(order)))[0]
mm = meta.set_index('StationId'); vals = rain[sel]
df = pd.DataFrame({'StationId': sid[sel], 'Latitude': mm.Latitude[sid[sel]].values, 'Longitude': mm.Longitude[sid[sel]].values,
                   'Year': year[sel], 'Month': month[sel], 'Record Type(1 or 2)': '', 'Number of Days': d['NumberOfDays'][:].data[sel]})
df['Mean_Row'] = [(round(float(np.nanmean(r)), 3) if np.isfinite(r).any() else np.nan) for r in vals]
for k in range(31): df[str(k + 1)] = vals[:, k]
df['District'] = ''; df['Station'] = mm.Station[sid[sel]].values; df['Classification'] = ''
df = df[df.Mean_Row.notna() | True][cols].sort_values(['StationId', 'Year', 'Month'])
df.to_csv(OUT / 'rf40_layout_filled_all_years.csv', index=True)
df[df.Year >= 2018].reset_index(drop=True).to_csv(OUT / 'rf40_layout_filled_2018_2025.csv', index=True)
print('csv rows', len(df))
Displaying build_station_gt.py.
