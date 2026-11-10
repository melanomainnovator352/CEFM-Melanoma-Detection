from pymongo import MongoClient
from app.core.config import MONGODB_URL, MONGODB_DATABASE


# ============================================================
# MONGODB CONNECTION
# ============================================================

client = MongoClient(
    MONGODB_URL,
    serverSelectionTimeoutMS=1000,
)

db = client[MONGODB_DATABASE]


# ============================================================
# COLLECTIONS
# ============================================================

cases_collection = db["cases"]
analyses_collection = db["analyses"]
rag_documents_collection = db["rag_documents"]


# ============================================================
# CONNECTION CHECK
# ============================================================

import os

def check_mongodb():
    """
    Check whether MongoDB is reachable.
    """
    if os.environ.get("TESTING") == "1":
        return False
    try:
        client.admin.command("ping")
        return True
    except Exception:
        return False