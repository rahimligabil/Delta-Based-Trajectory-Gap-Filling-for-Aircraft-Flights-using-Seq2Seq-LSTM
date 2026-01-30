import os
import csv
import math
import time as pytime
import pandas as pd

try:
    import trino
    from trino.exceptions import TrinoAuthError, HttpError
except ImportError as exc:
    raise SystemExit(
        "Python package 'trino' is not installed. "
        "Run `pip install -r requirements.txt` (or `pip install trino`) first."
    ) from exc

# =========================
# 1) SETTINGS
# =========================

TRINO_HOST = "trino.opensky-network.org"
TRINO_PORT = 443
TRINO_USER = "USERNAME"       # <- fill this
TRINO_PASSWORD = "PASSWORD"   # <- fill this (only for basic auth)
# Auth type:
# - "oauth": Uses Trino external-auth (--external-authentication) flow, same as the CLI.
# - "basic": Username/password basic auth.
TRINO_AUTH_MODE = "oauth"

TRINO_CATALOG = "minio"
TRINO_SCHEMA = "osky"

STATE_TABLE = "state_vectors_data4"

# Sampling: 1 point every 5 seconds. Set to 1 to sample every second (files grow quickly).
SAMPLE_EVERY_N_SECONDS = 5

# How many rows to fetch per call? (keeps RAM in check)
FETCH_CHUNK_SIZE = 5000

# Files
FLIGHTS_CSV_PATH = r"data/flights_test.csv"
OUT_DIR = r"state_vectors/test"

# To allow resuming a partial download:
SKIP_IF_EXISTS = True


# =========================
# 2) TRINO CONNECTION
# =========================

def make_trino_conn():
    """
    TRINO_AUTH_MODE = oauth => same flow as CLI --external-authentication.
    TRINO_AUTH_MODE = basic => username/password basic auth.
    """
    if TRINO_AUTH_MODE == "basic":
        auth_obj = trino.auth.BasicAuthentication(TRINO_USER, TRINO_PASSWORD)
    elif TRINO_AUTH_MODE == "oauth":
        auth_obj = trino.auth.OAuth2Authentication()
    else:
        raise SystemExit(f"Unknown TRINO_AUTH_MODE: {TRINO_AUTH_MODE}")

    conn = trino.dbapi.connect(
        host=TRINO_HOST,
        port=TRINO_PORT,
        user=TRINO_USER,
        http_scheme="https",
        auth=auth_obj,
        catalog=TRINO_CATALOG,
        schema=TRINO_SCHEMA,
    )
    return conn


def assert_trino_login():
    """
    Test login with a simple SELECT 1.
    If it returns 401, username/password or access permissions are wrong.
    """
    try:
        conn = make_trino_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        return conn
    except (TrinoAuthError, HttpError) as exc:
        status = exc.args[0] if exc.args else None
        if status == 401:
            raise SystemExit(
                "Trino 401: Server rejected username/password. "
                "Verify that you can log in with your OpenSky account and that it has Trino access."
            ) from exc
        raise


# =========================
# 3) FETCH STATES FOR ONE FLIGHT
# =========================

def download_one_flight(conn, flight_id, icao24, firstseen, lastseen, out_path):
    """
    Pull state vector rows for a single flight from Trino and write them to a CSV file.

    Why this approach?
    - Filter by the flight's time window (firstseen..lastseen).
    - Filter by icao24 to avoid rows from other aircraft in the same window.
    - Add an hour partition filter so Trino scans less data.
    - Reduce row count with SAMPLE_EVERY_N_SECONDS.
    - Fetch in chunks with fetchmany to avoid blowing up RAM.
    """

    # hour partition range (epoch hour start)
    hour_start = (int(firstseen) // 3600) * 3600
    hour_end = (int(lastseen) // 3600) * 3600

    # SQL: select only the columns we need
    # (more columns -> larger files and more IO)
    sql = f"""
    SELECT
        time,
        lat,
        lon,
        geoaltitude,
        velocity,
        heading,
        vertrate,
        onground
    FROM {STATE_TABLE}
    WHERE icao24 = '{icao24}'
      AND time BETWEEN {int(firstseen)} AND {int(lastseen)}
      AND hour BETWEEN {hour_start} AND {hour_end}
      AND (time % {SAMPLE_EVERY_N_SECONDS} = 0)
    ORDER BY time
    """

    cur = conn.cursor()
    cur.execute(sql)

    # Stream-write to file
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "lat", "lon", "geoaltitude", "velocity", "heading", "vertrate", "onground"])

        total_rows = 0
        while True:
            rows = cur.fetchmany(FETCH_CHUNK_SIZE)
            if not rows:
                break
            writer.writerows(rows)
            total_rows += len(rows)

    return total_rows


# =========================
# 4) MAIN FLOW: flights_train.csv -> state_vectors/train/*
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # read flights_train.csv
    df = pd.read_csv(FLIGHTS_CSV_PATH)

    # Expected columns:
    # day,icao24,callsign,firstseen,lastseen,duration_sec,flight_id,split
    required = {"flight_id", "icao24", "firstseen", "lastseen"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    conn = assert_trino_login()

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for i, row in df.iterrows():
        flight_id = str(row["flight_id"]).strip()
        icao24 = str(row["icao24"]).strip()
        firstseen = int(row["firstseen"])
        lastseen = int(row["lastseen"])

        out_path = os.path.join(OUT_DIR, f"{flight_id}.csv")

        if SKIP_IF_EXISTS and os.path.exists(out_path):
            skip_count += 1
            continue

        try:
            t0 = pytime.time()
            n = download_one_flight(conn, flight_id, icao24, firstseen, lastseen, out_path)
            dt = pytime.time() - t0

            ok_count += 1
            print(f"[OK] {ok_count} | flight_id={flight_id} rows={n} time={dt:.1f}s")

        except (TrinoAuthError, HttpError) as e:
            status = e.args[0] if e.args else None
            if status == 401:
                print(f"[FAIL] flight_id={flight_id} err=401 Unauthorized (username/password rejected)")
                raise SystemExit(
                    "Trino 401: Username/password was rejected. "
                    "Check your credentials and that your account has Trino access."
                ) from e
            fail_count += 1
            print(f"[FAIL] flight_id={flight_id} err={e}")
        except Exception as e:
            fail_count += 1
            print(f"[FAIL] flight_id={flight_id} err={e}")

            # Here you could delete the bad file, retry, etc.
            # For now we just log it.

    print("\nDONE")
    print(f"OK   : {ok_count}")
    print(f"SKIP : {skip_count}")
    print(f"FAIL : {fail_count}")


if __name__ == "__main__":
    main()
