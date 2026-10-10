import os
import re
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


# =========================================================
# APP
# =========================================================

app = FastAPI(title="FoodVista AI")

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATASET PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "FoodVista_App_Dataset_VERIFIED_UPDATED.csv",
)


# =========================================================
# LOAD DATASET
# =========================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Please place FoodVista_App_Dataset_VERIFIED_UPDATED.csv "
        "inside the dataset folder."
    )

df = pd.read_csv(
    DATASET_PATH,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean(value):
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in {
        "nan", "none", "null", "nat", "n/a", "na", "-"
    }:
        return ""

    return value


def normalize(value):
    value = clean(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def valid_url(value):
    value = clean(value)

    if not value:
        return False

    return value.lower().startswith(("https://", "http://"))


def get_value(row, column):
    return clean(row.get(column, ""))


# =========================================================
# REQUIRED DATASET COLUMNS
# =========================================================

required_columns = [
    "Restaurant",
    "Swiggy URL",
    "Zomato URL",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns missing from dataset: "
        + ", ".join(missing_columns)
    )


# =========================================================
# CREATE SEARCH TEXT
# =========================================================

def make_text(row):
    fields = [
        row.get("Restaurant", ""),
        row.get("Cuisine", ""),
        row.get("Swiggy Area", ""),
        row.get("Swiggy Location", ""),
        row.get("Zomato Restaurant", ""),
        row.get("Zomato Locality", ""),
        row.get("Zomato Address", ""),
    ]

    return " ".join(
        normalize(value)
        for value in fields
        if clean(value)
    )


df["_search_text"] = df.apply(make_text, axis=1)

# =========================================================
# NEW CITIES DATASET (Vizag kaadu, migata cities)
# =========================================================

NEW_DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "FoodVista_App_Dataset_NEWCITIES.csv",
)

if os.path.exists(NEW_DATASET_PATH):
    new_df = pd.read_csv(
        NEW_DATASET_PATH,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    new_df["_search_text"] = new_df.apply(make_text, axis=1)
    new_df["_city_n"] = new_df["Swiggy Location"].apply(normalize)
else:
    new_df = pd.DataFrame(columns=list(df.columns) + ["_city_n"])


@app.get("/cities")
def get_cities():
    others = [
        c for c in new_df["Swiggy Location"].apply(clean).value_counts().index
        if c and normalize(c) not in {"vizag", "visakhapatnam", "vishakhapatnam"}
    ]
    return {"cities": ["Vizag"] + sorted(others)}


# =========================================================
# VALID COMPARISONS
# Only show rows with usable Swiggy AND Zomato URLs.
# Do not modify or delete the original dataframe.
# =========================================================

def is_valid_comparison(row):
    swiggy_url = get_value(row, "Swiggy URL")
    zomato_url = get_value(row, "Zomato URL")

    swiggy_name = get_value(row, "Restaurant")
    zomato_name = get_value(row, "Zomato Restaurant")

    if not valid_url(swiggy_url):
        return False

    if not valid_url(zomato_url):
        return False

    if not swiggy_name or not zomato_name:
        return False

    return True


valid_df = df[
    df.apply(is_valid_comparison, axis=1)
].copy()

# Remove exact duplicate comparison records only.
valid_df = valid_df.drop_duplicates(
    subset=[
        "Restaurant",
        "Zomato Restaurant",
        "Swiggy URL",
        "Zomato URL",
    ]
).reset_index(drop=True)


# =========================================================
# CITY FILTER
# Dataset is intended for Vizag restaurant comparisons.
# =========================================================

def city_ok(row, city):
    city = normalize(city)

    if not city or city in {
        "all", "all cities", "any"
    }:
        return True

    if city in {
        "vizag",
        "visakhapatnam",
        "vishakhapatnam",
    }:
        return True

    location_text = " ".join([
        normalize(row.get("Swiggy Area", "")),
        normalize(row.get("Swiggy Location", "")),
        normalize(row.get("Zomato Locality", "")),
        normalize(row.get("Zomato Address", "")),
    ])

    return city in location_text


# =========================================================
# CREATE RESULT
# Preserve existing frontend response field names.
# =========================================================

def result_from_row(row):
    return {
        "restaurant_name": get_value(
            row, "Restaurant"
        ),

        "cuisine": get_value(
            row, "Cuisine"
        ),

        "average_price": get_value(
            row, "Average Price"
        ),

        "number_of_offers": get_value(
            row, "Number of Offers"
        ),

        "offer_name": get_value(
            row, "Offer Name"
        ),

        "pure_veg": get_value(
            row, "Pure Veg"
        ),

        "swiggy": {
            "rating": get_value(
                row, "Swiggy Rating"
            ),

            "rating_count": get_value(
                row, "Swiggy Rating Count"
            ),

            "area": get_value(
                row, "Swiggy Area"
            ),

            "location": get_value(
                row, "Swiggy Location"
            ),

            "url": get_value(
                row, "Swiggy URL"
            ),

            "url_status": get_value(
                row, "Swiggy URL Match Status"
            ) or "available",
        },

        "zomato": {
            "restaurant_name": get_value(
                row, "Zomato Restaurant"
            ),

            "rating": get_value(
                row, "Zomato Rating"
            ),

            "rating_count": get_value(
                row, "Zomato Rating Count"
            ),

            "locality": get_value(
                row, "Zomato Locality"
            ),

            "address": get_value(
                row, "Zomato Address"
            ),

            "url": get_value(
                row, "Zomato URL"
            ),

            "avg_item_price": get_value(
                row, "Zomato Avg Item Price"
            ),
        },

        "match_score": get_value(
            row, "Match Score"
        ),

        "clean_decision": get_value(
            row, "Clean Decision"
        ),
    }


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "FoodVista AI backend is running",
        "dataset": os.path.basename(DATASET_PATH),
        "total_records": len(df),
        "valid_comparisons": len(valid_df),
    }


# =========================================================
# DATASET STATISTICS
# =========================================================

@app.get("/dataset-stats")
def dataset_stats():
    swiggy_urls = df["Swiggy URL"].apply(clean)
    zomato_urls = df["Zomato URL"].apply(clean)

    return {
        "dataset": os.path.basename(DATASET_PATH),

        "total_records": len(df),

        "valid_comparisons": len(valid_df),

        "comparisons_hidden": len(df) - len(valid_df),

        "swiggy_urls_available": int(
            swiggy_urls.apply(valid_url).sum()
        ),

        "swiggy_urls_missing": int(
            (~swiggy_urls.apply(valid_url)).sum()
        ),

        "zomato_urls_available": int(
            zomato_urls.apply(valid_url).sum()
        ),

        "zomato_urls_missing": int(
            (~zomato_urls.apply(valid_url)).sum()
        ),

        "note": (
            "Only comparisons with both platform URLs and "
            "both restaurant names are returned by search."
        ),
    }



# =========================================================
# SEARCH ENDPOINT
# =========================================================

@app.get("/search")
def search(
    city: str = Query(default="Vizag"),
    query: str = Query(default=""),
):
    q = normalize(query)

    # Preserve existing Vizag behavior.
    is_vizag = normalize(city) in {
        "vizag",
        "visakhapatnam",
        "vishakhapatnam",
    }

    if is_vizag:
        # Vizag: paatha logic, em marchaledu.
        working = valid_df[
            valid_df.apply(
                lambda row: city_ok(row, city),
                axis=1,
            )
        ].copy()
    else:
        # Migata cities: kotha dataset, URLs mandatory kaadu.
        working = new_df[
            new_df["_city_n"] == normalize(city)
        ].copy()

    if q:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", q)
            if len(token) > 1
        ]

        def score(row):
            restaurant = normalize(
                row.get("Restaurant", "")
            )
            zomato_name = normalize(
                row.get("Zomato Restaurant", "")
            )
            search_text = row.get("_search_text", "")

            if q == restaurant or q == zomato_name:
                return 100

            if q in restaurant or q in zomato_name:
                return 75

            if q in search_text:
                return 50

            if tokens and all(
                token in search_text
                for token in tokens
            ):
                return 10 + len(tokens)

            return 0

        working["_score"] = working.apply(score, axis=1)
        matches = working[working["_score"] > 0].copy()
        matches = matches.sort_values(
            by=["_score", "Restaurant"],
            ascending=[False, True],
        )
    else:
        matches = working.sort_values(
            by=["Restaurant"],
            ascending=True,
        )

    matches = matches.head(100)

    results = [
        result_from_row(row)
        for _, row in matches.iterrows()
    ]
    # Kotha cities (Vizag kaadu): Swiggy and Zomato links hide
    if not is_vizag:
        for r in results:
            r["swiggy"]["url"] = ""
            r["zomato"]["url"] = ""
            r["swiggy"]["url_status"] = ""

    return {
        "count": len(results),
        "city": city,
        "query": query,
        "dataset": os.path.basename(DATASET_PATH),
        "dataset_records": len(df),
        "valid_comparisons": len(valid_df),
        "results": results,
    }


# =========================================================
# LIVE SEARCH ENDPOINT
# Compatibility endpoint for existing frontend.
# This searches local CSV data, not live platform APIs.
# =========================================================

@app.get("/live-search")
def live_search(
    query: str = Query(default="")
):
    return search(
        city="Vizag",
        query=query,
    )

@app.get("/app")
def frontend_page():
    return FileResponse(
        os.path.join(BASE_DIR, "frontend", "index.html")
    )