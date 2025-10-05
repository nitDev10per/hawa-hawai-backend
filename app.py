from flask import Flask, request, jsonify
import pandas as pd
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

def fetch_data(lat, lon, target_date, start_year, end_year, parameters, window=5):
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    if window:
        date_start = target_dt - timedelta(days=window)
        date_end = target_dt + timedelta(days=window)
    else:
        date_start = target_dt
        date_end = target_dt

    month_start = date_start.month
    day_start = date_start.day
    month_end = date_end.month
    day_end = date_end.day

    all_records = []
    param_str = ",".join(parameters)

    for year in range(start_year, end_year + 1):
        try:
            start_range = datetime(year, month_start, day_start).strftime('%Y%m%d')
            end_range = datetime(year, month_end, day_end).strftime('%Y%m%d')
        except ValueError:
            continue  # Skip invalid dates (e.g., Feb 29 on non-leap years)

        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters={param_str}"
            f"&community=RE&longitude={lon}&latitude={lat}"
            f"&start={start_range}&end={end_range}&format=JSON&units=metric&header=true&time-standard=utc"
        )
        headers = {'accept': 'application/json'}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            try:
                lon_, lat_, elev_ = data['geometry']['coordinates']
                params = data['properties']['parameter']
                
                dates = list(params[parameters[0]].keys())
                for d in dates:
                    record = {
                        'Longitude': lon_,
                        'Latitude': lat_,
                        'Elevation': elev_,
                        'Date': pd.to_datetime(d, format='%Y%m%d')
                    }
                    missing = False
                    for p in parameters:
                        val = params[p][d]
                        record[p] = val
                        if val == -999.00:
                            missing = True
                    if not missing:
                        all_records.append(record)
            except KeyError:
                # In case json does not contain expected keys
                continue
        else:
            print(f"Failed to fetch data for {start_range} to {end_range}: HTTP {response.status_code}")

    return pd.DataFrame(all_records)


def categorize_aod(aod):
    if aod < 0.15:
        return "Clean"
    elif 0.15 <= aod < 0.40:
        return "Moderate"
    elif 0.40 <= aod < 0.80:
        return "Heavily Polluted"
    else:
        return "Extremely Polluted"

def categorize_cloud(ca):
    if ca < 30:
        return "Sunny"
    elif ca > 70:
        return "Cloudy"
    else:
        return "Partly Cloudy"

def categorize_temp(t_celsius):
    if t_celsius <= -10:  # <= -10°C
        return "Extremely Cold (<= -10°C)"
    elif -10 < t_celsius <= 0:  # -10°C to 0°C
        return "Very Cold (-10°C to 0°C)"
    elif 0 < t_celsius <= 10:  # 0°C to 10°C
        return "Cold (0°C to 10°C)"
    elif 10 < t_celsius <= 20:  # 10°C to 20°C
        return "Mild (10°C to 20°C)"
    elif 20 < t_celsius <= 35:  # 20°C to 35°C
        return "Warm (20°C to 35°C)"
    elif 35 < t_celsius <= 45:  # 35°C to 45°C
        return "Hot (35°C to 45°C)"
    else:
        return "Extremely Hot (> 45°C)"



def categorize_snow(s):
    if s == 0:
        return 'No Snow'
    elif s < 1:
        return 'Light Snow'
    elif s < 5:
        return 'Moderate Snow'
    else:
        return 'Heavy Snow'

def categorize_rainfall(x):
    if x == 0:
        return 'No Rain'
    elif x < 5:
        return 'Light Rain'
    elif x < 20:
        return 'Moderate Rain'
    else:
        return 'Heavy Rain'

def categorize_wind(ws):
    if ws < 2:
        return 'Calm'
    elif ws < 5:
        return 'Light Breeze'
    elif ws < 10:
        return 'Moderate Breeze'
    else:
        return 'Strong Wind'

@app.route('/api/aod')
def api_aod():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('long'))
    target_date = request.args.get('date')
    start_year = int(request.args.get('start_year', 2000))
    end_year = int(request.args.get('end_year', 2025))
    df = fetch_data(lat, lon, target_date, start_year, end_year, ['AOD_55_ADJ'])
    df['Category'] = df['AOD_55_ADJ'].apply(categorize_aod)
    probs = df['Category'].value_counts(normalize=True) * 100
    return jsonify(probs.to_dict())

@app.route('/api/cloud')
def api_cloud():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('long'))
    target_date = request.args.get('date')
    start_year = int(request.args.get('start_year', 2000))
    end_year = int(request.args.get('end_year', 2025))
    df = fetch_data(lat, lon, target_date, start_year, end_year, ['CLOUD_AMT'])
    df['Category'] = df['CLOUD_AMT'].apply(categorize_cloud)
    probs = df['Category'].value_counts(normalize=True) * 100
    return jsonify(probs.to_dict())

@app.route('/api/temp')
def api_temp():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('long'))
    target_date = request.args.get('date')
    start_year = int(request.args.get('start_year', 2000))
    end_year = int(request.args.get('end_year', 2025))
    df = fetch_data(lat, lon, target_date, start_year, end_year, ['T2M'])
    df['Category'] = df['T2M'].apply(categorize_temp)
    probs = df['Category'].value_counts(normalize=True) * 100
    return jsonify(probs.to_dict())

@app.route('/api/snow')
def api_snow():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('long'))
    target_date = request.args.get('date')
    start_year = int(request.args.get('start_year', 2000))
    end_year = int(request.args.get('end_year', 2025))
    df = fetch_data(lat, lon, target_date, start_year, end_year, ['SNODP'])
    df['Category'] = df['SNODP'].apply(categorize_snow)
    probs = df['Category'].value_counts(normalize=True) * 100
    return jsonify(probs.to_dict())

@app.route('/api/rain')
def api_rain():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('long'))
    target_date = request.args.get('date')
    start_year = int(request.args.get('start_year', 2000))
    end_year = int(request.args.get('end_year', 2025))
    df = fetch_data(lat, lon, target_date, start_year, end_year, ['PRECTOTCORR'])
    df['Category'] = df['PRECTOTCORR'].apply(categorize_rainfall)
    probs = df['Category'].value_counts(normalize=True) * 100
    return jsonify(probs.to_dict())

@app.route('/api/wind')
def api_wind():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('long'))
    target_date = request.args.get('date')
    start_year = int(request.args.get('start_year', 2000))
    end_year = int(request.args.get('end_year', 2025))
    df = fetch_data(lat, lon, target_date, start_year, end_year, ['WS10M'])
    df['Category'] = df['WS10M'].apply(categorize_wind)
    probs = df['Category'].value_counts(normalize=True) * 100
    return jsonify(probs.to_dict())


@app.route('/api/timeseries')
def api_timeseries():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('long'))
    target_date = request.args.get('date')
    start_year = int(request.args.get('start_year', 2000))
    end_year = int(request.args.get('end_year', 2025))
    params = request.args.get('parameters', 'AOD_55_ADJ,CLOUD_AMT,T2M,SNODP,PRECTOTCORR,WS10M')
    parameter_list = [p.strip() for p in params.split(',')]
    df = fetch_data(lat, lon, target_date, start_year, end_year, parameter_list, window=0)
    result = df.to_dict(orient='records')
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
