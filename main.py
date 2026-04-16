import os
import time
import json
import asyncio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from google.cloud import bigquery
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel

# --- 1. Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "maven-search-493109")
REGION = os.environ.get("GCP_REGION", "us-central1")
# Your single table that contains both vectors AND metadata
VECTOR_TABLE = f"{PROJECT_ID}.slikk_data.maven_final_vector_search"

# --- 2. Initialize Clients ---
vertexai.init(project=PROJECT_ID, location=REGION)
mm_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)
bq_client = bigquery.Client(project=PROJECT_ID)

# --- 3. Constants ---
VALID_DIVISIONS = [
    'beauty', 'footwear', "women's jewellery", 'women', 'men',
    'accessories/men', 'footwear/women', 'travel & luggages',
    'accessories/women', 'home', 'accessories', 'footwear/men', 'men/women'
]

GENDER_MAP = {
    "men":   ["men", "men/women", "footwear/men", "accessories/men"],
    "women": ["women", "men/women", "women's jewellery", "footwear/women", "accessories/women"],
}

app = FastAPI(title="MAVEN UNIVERSAL SEARCH API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- ASYNC TASKS ---
async def async_parse_user_query(raw_query: str):
    prompt = f"""
    Analyze search query: "{raw_query}"
    Tasks: 1. Fix spelling. 2. Extract Color. 3. Extract Product Type. 4. Detect Gender. 
    5. Categorize into division from: {json.dumps(VALID_DIVISIONS)}
    Return ONLY JSON with keys: corrected_query, color, product_type, gender, division.
    """
    try:
        response = genai_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
        )
        return json.loads(response.text.strip())
    except:
        return {"corrected_query": raw_query}

async def async_get_embedding(text: str):
    # Using the verified 1408-dimension logic
    embeddings = mm_model.get_embeddings(contextual_text=text, dimension=1408)
    return [float(x) for x in embeddings.text_embedding]

# --- SEARCH ENDPOINT ---
@app.get("/search")
async def search(query: str, page: int = 1, page_size: int = 12):
    start_time = time.time()
    offset = (page - 1) * page_size

    try:
        parsed_filters, query_vector = await asyncio.gather(
            async_parse_user_query(query), 
            async_get_embedding(query)
        )

        # Build dynamic filters using the 'base.' prefix
        where_clauses = ["1=1"]
        query_params = [
            bigquery.ArrayQueryParameter("query_vector", "FLOAT64", query_vector),
            bigquery.ScalarQueryParameter("limit", "INT64", page_size + 1),
            bigquery.ScalarQueryParameter("offset", "INT64", offset),
        ]

        if parsed_filters.get("gender") and parsed_filters["gender"].lower() in GENDER_MAP:
            divs = GENDER_MAP[parsed_filters["gender"].lower()]
            where_clauses.append("LOWER(base.`Division Name`) IN UNNEST(@filter_divisions)")
            query_params.append(bigquery.ArrayQueryParameter("filter_divisions", "STRING", [d.lower() for d in divs]))
        
        if parsed_filters.get("color"):
            where_clauses.append("LOWER(base.color) LIKE @color")
            query_params.append(bigquery.ScalarQueryParameter("color", "STRING", f"%{parsed_filters['color'].lower()}%"))

        # The working SQL structure we just tested
        sql_query = f"""
            SELECT 
                base.skid AS product_id, 
                base.`Product Name`, 
                base.`Brand Name`, 
                base.`Division Name`,
                base.`Color code swatch link` AS image_url, 
                base.SP AS price, 
                distance
            FROM VECTOR_SEARCH(
                TABLE `{VECTOR_TABLE}`, 
                'embedding',
                (SELECT @query_vector AS embedding),
                top_k => 200
            )
            WHERE {" AND ".join(where_clauses)}
            ORDER BY distance ASC
            LIMIT @limit OFFSET @offset
        """

        print(sql_query)

        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        query_job = bq_client.query(sql_query, job_config=job_config)
        rows = list(query_job.result())

        results = [{
            "product_id": r.product_id,
            "name": r.get("Product Name"),
            "brand": r.get("Brand Name"),
            "price": float(r.get("price", 0)) if r.get("price") else 0.0,
            "image_url": r.get("image_url"),
            "relevance": round(1.0 - float(r.distance), 4)
        } for r in rows[:page_size]]

        return {
            "query": query,
            "query_time_ms": int((time.time() - start_time) * 1000),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))
