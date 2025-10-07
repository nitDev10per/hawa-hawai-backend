from flask import Flask, json, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS globally

# -----------------------------
# Fetch Data Function (timeout-safe)
# -----------------------------
def fetch_data(lat, lon, target_date, start_year, end_year, parameters, window=5):
    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Expected YYYY-MM-DD")

    if window:
        date_start = target_dt - timedelta(days=window)
        date_end = target_dt + timedelta(days=window)
    else:
        date_start = target_dt
        date_end = target_dt

    month_start, day_start = date_start.month, date_start.day
    month_end, day_end = date_end.month, date_end.day

    all_records = []
    param_str = ",".join(parameters)

    for year in range(start_year, end_year + 1):
        try:
            start_range = datetime(year, month_start, day_start).strftime('%Y%m%d')
            end_range = datetime(year, month_end, day_end).strftime('%Y%m%d')
        except ValueError:
            continue  # skip invalid dates (Feb 29 on non-leap years)

        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters={param_str}"
            f"&community=RE&longitude={lon}&latitude={lat}"
            f"&start={start_range}&end={end_range}"
            f"&format=JSON&units=metric&header=true&time-standard=utc"
        )

        try:
            # ⚡ Timeout set to 15 seconds
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            print(f"[WARNING] Timeout for year {year}, skipping...")
            continue
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed for year {year}: {e}")
            continue
        except ValueError:
            print(f"[ERROR] JSON decode failed for year {year}")
            continue

        # Parse the data safely
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
                    val = params.get(p, {}).get(d, -999.00)
                    record[p] = val
                    if val == -999.00:
                        missing = True
                if not missing:
                    all_records.append(record)
        except Exception as e:
            print(f"[ERROR] Failed to parse data for year {year}: {e}")
            continue

    df = pd.DataFrame(all_records)
    if df.empty:
        raise ValueError("No valid data returned from NASA POWER API.")
    return df

def fetch_params(lat, lon, target_date, start_year, end_year, parameters, window=5):
    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Expected YYYY-MM-DD")

    if window:
        date_start = target_dt - timedelta(days=window)
        date_end = target_dt + timedelta(days=window)
    else:
        date_start = target_dt
        date_end = target_dt

    month_start, day_start = date_start.month, date_start.day
    month_end, day_end = date_end.month, date_end.day

    df = {
        "start_year": start_year,
        "end_year": end_year,
        "month_start": month_start,
        "day_start": day_start,
        "month_end": month_end,
        "day_end": day_end,
        "parameters": parameters,
        "lat": lat,
        "lon": lon
    }
    return df


def get_request_params():
    validate_request_params('lat', 'long', 'date')
    lat = float(request.args['lat'])
    lon = float(request.args['long'])
    target_date = request.args['date']
    start_year = int(request.args.get('start_year', 2000))
    end_year = int(request.args.get('end_year', 2025))
    return lat, lon, target_date, start_year, end_year

def end_response(categories_fun, params):
    # Get JSON data from POST body
    data = request.get_json()
    if not data or 'api_result' not in data:
        return jsonify({"error": "Missing 'api_result' in request body"}), 400

    # Extract and parse data
    df_json = data['api_result']
    if isinstance(df_json, str):
        # Handle case where it's a JSON string
        df_json = json.loads(df_json)

    # Convert to DataFrame
    df = pd.DataFrame(df_json)

    # Apply category function
    df['Category'] = df[params].apply(categories_fun)

    # Calculate probabilities
    probs = df['Category'].value_counts(normalize=True) * 100

    # Return result as JSON
    return jsonify(probs.to_dict())
# ---------------------------------------------------
# Categorization Functions
# ---------------------------------------------------
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
    if t_celsius <= -10:
        return "Extremely Cold (<= -10°C)"
    elif -10 < t_celsius <= 0:
        return "Very Cold (-10°C to 0°C)"
    elif 0 < t_celsius <= 10:
        return "Cold (0°C to 10°C)"
    elif 10 < t_celsius <= 20:
        return "Mild (10°C to 20°C)"
    elif 20 < t_celsius <= 35:
        return "Warm (20°C to 35°C)"
    elif 35 < t_celsius <= 45:
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


# ---------------------------------------------------
# Utility: Validate required parameters
# ---------------------------------------------------
def validate_request_params(*params):
    missing = [p for p in params if request.args.get(p) is None]
    if missing:
        raise ValueError(f"Missing required query parameters: {', '.join(missing)}")


# ---------------------------------------------------
# API Endpoints
# ---------------------------------------------------

@app.route('/api/aod')
def api_aod():
    try:
        lat, lon, target_date, start_year, end_year = get_request_params()
        df = fetch_params(lat, lon, target_date, start_year, end_year, ['AOD_55_ADJ'])
        return jsonify(df)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/aod_after_res', methods=['POST'])
def api_aod_after_res():
    try:
        return end_response(categorize_aod, 'AOD_55_ADJ')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/cloud')
def api_cloud():
    try:
        lat, lon, target_date, start_year, end_year = get_request_params()
        df = fetch_params(lat, lon, target_date, start_year, end_year, ['CLOUD_AMT'])
        return jsonify(df)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/cloud_after_res', methods=['POST'])
def api_cloud_after_res():
    try:
        return end_response(categorize_cloud, 'CLOUD_AMT')
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/temp')
def api_temp():
    try:
        lat, lon, target_date, start_year, end_year = get_request_params()
        df = fetch_params(lat, lon, target_date, start_year, end_year, ['T2M'])
        return jsonify(df)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/temp_after_res', methods=['POST'])
def api_temp_after_res():
    try:
        return end_response(categorize_temp, 'T2M')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/snow')
def api_snow():
    try:
        lat, lon, target_date, start_year, end_year = get_request_params()
        df = fetch_params(lat, lon, target_date, start_year, end_year, ['SNODP'])
        return jsonify(df)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/snow_after_res', methods=['POST'])
def api_snow_after_res():
    try:
        return end_response(categorize_snow, 'SNODP')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/rain')
def api_rain():
    try:
        lat, lon, target_date, start_year, end_year = get_request_params()
        df = fetch_params(lat, lon, target_date, start_year, end_year, ['PRECTOTCORR'])
        return jsonify(df)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/rain_after_res', methods=['POST'])
def api_rain_after_res():
    try:
        return end_response(categorize_rainfall, 'PRECTOTCORR')
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/wind')
def api_wind():
    try:
        lat, lon, target_date, start_year, end_year = get_request_params()
        df = fetch_params(lat, lon, target_date, start_year, end_year, ['WS10M'])
        return jsonify(df)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/wind_after_res', methods=['POST'])
def api_wind_after_res():
    try:
        return end_response(categorize_wind, 'WS10M')
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/api/timeseries')
def api_timeseries():
    try:
        validate_request_params('lat', 'long', 'date')
        lat = float(request.args['lat'])
        lon = float(request.args['long'])
        target_date = request.args['date']
        start_year = int(request.args.get('start_year', 2000))
        end_year = int(request.args.get('end_year', 2025))
        params = request.args.get('parameters', 'AOD_55_ADJ,CLOUD_AMT,T2M,SNODP,PRECTOTCORR,WS10M')
        parameter_list = [p.strip() for p in params.split(',')]

        df = fetch_data(lat, lon, target_date, start_year, end_year, parameter_list, window=0)
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------------------------------
# Global Error Handler
# ---------------------------------------------------
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500


# ---------------------------------------------------
# Run App
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
