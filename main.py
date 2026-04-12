import os
import time
import json
import asyncio
import configparser
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from google.cloud import bigquery

# --- 1. Configuration & Global Clients ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "maven-search-493109")
REGION = os.environ.get("GCP_REGION", "us-central1")
BQ_TABLE = f"{PROJECT_ID}.slikk_data.maven_final_vector_search"

# Initialize Clients
client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)
bq_client = bigquery.Client(project=PROJECT_ID)

# --- 2. Constants & Mapping ---
VALID_DIVISIONS = [
    'beauty', 'footwear', "women's jewellery", 'women', 'men',
    'accessories/men', 'footwear/women', 'travel & luggages',
    'accessories/women', 'home', 'accessories', 'footwear/men', 'men/women'
]

GENDER_MAP = {
    "men":   ["men", "men/women", "footwear/men", "accessories/men"],
    "women": ["women", "men/women", "women's jewellery", "footwear/women", "accessories/women"],
}

# --- 3. FastAPI Setup ---
app = FastAPI(title="MAVEN UNIVERSAL SEARCH API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ASYNC TASK 1: AI QUERY PARSING ---
async def async_parse_user_query(raw_query: str):
    prompt = f"""
    Analyze search query: "{raw_query}"
    Tasks: 1. Fix spelling. 2. Extract Color. 3. Extract Product Type. 4. Detect Gender (men/women). 5. Extract Brand. 
    6. Extract price_min/price_max. 7. Categorize into division from: {json.dumps(VALID_DIVISIONS)}
    8. Generate corrected_query (cleaned phrase).

    Return ONLY raw JSON with keys: corrected_query, color, product_type, gender, division, brand, price_min, price_max.
    """
    try:
        # Note: In the genai SDK, we use models.generate_content
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
        )
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"Parsing Failed: {e}")
        return {"corrected_query": raw_query, "color": None, "gender": None, "brand": None}

# --- ASYNC TASK 2: VECTOR GENERATION ---
async def async_get_embedding(text: str):
    # We wrap the text in a Content object to ensure the SDK sends it correctly
    response = client.models.embed_content(
        model="multimodalembedding@001",
        contents=types.Content(parts=[types.Part(text=text)])
    )
    # Extract the vector values
    return [float(x) for x in response.embeddings[0].values]

# --- THE SEARCH ENDPOINT ---
@app.get("/search")
async def search_catalogue(query: str, page: int = 1, page_size: int = 12):
    start_time = time.time()
    offset = (page - 1) * page_size
    
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Parallel Execution
        parsed_filters, query_vector = await asyncio.gather(
            async_parse_user_query(query), 
            async_get_embedding(query)
        )

        # Dynamic SQL Construction
        where_clauses = ["1=1"]
        query_params = [
            bigquery.ArrayQueryParameter("query_vector", "FLOAT64", query_vector),
            bigquery.ScalarQueryParameter("limit", "INT64", page_size + 1),
            bigquery.ScalarQueryParameter("offset", "INT64", offset),
        ]

        if parsed_filters.get("gender") and parsed_filters["gender"].lower() in GENDER_MAP:
            allowed_divs = GENDER_MAP[parsed_filters["gender"].lower()]
            where_clauses.append("LOWER(`Division Name`) IN UNNEST(@filter_divisions)")
            query_params.append(bigquery.ArrayQueryParameter("filter_divisions", "STRING", [d.lower() for d in allowed_divs]))
        
        if parsed_filters.get("color"):
            where_clauses.append("LOWER(color) LIKE @color")
            query_params.append(bigquery.ScalarQueryParameter("color", "STRING", f"%{parsed_filters['color'].lower()}%"))

        where_sql = " AND ".join(where_clauses)

        sql_query = f"""
            SELECT 
                skid, `Product Name`, `Brand Name`, `Division Name`,
                `Color code swatch link` as image_url, SP as price, distance
            FROM VECTOR_SEARCH(
                TABLE `{BQ_TABLE}`, 
                'embedding',
                (SELECT @query_vector AS embedding),
                top_k => 500
            )
            WHERE {where_sql}
            ORDER BY distance ASC
            LIMIT @limit OFFSET @offset
        """

        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        query_job = bq_client.query(sql_query, job_config=job_config)
        rows = list(query_job.result())

        results = []
        for row in rows[:page_size]:
            results.append({
                "product_id": row.skid,
                "name": row.get("Product Name"),
                "brand": row.get("Brand Name"),
                "price": float(row.get("price", 0)) if row.get("price") else 0.0,
                "image_url": row.get("image_url"),
                "relevance": round(1.0 - float(row.distance), 4)
            })

        return {
            "query": query,
            "query_time_ms": int((time.time() - start_time) * 1000),
            "pagination": {"current_page": page, "has_next_page": len(rows) > page_size},
            "results": results
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Critical: Cloud Run passes the port as an environment variable
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
