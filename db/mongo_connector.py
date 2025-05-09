"""
MongoDB connector module for stock dashboard application.
Provides functions to connect to MongoDB and interact with collections.
"""
import logging
from typing import Dict, List, Optional, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure
import streamlit as st

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

@st.cache_resource
def get_mongo_client() -> MongoClient:
    """
    Get a cached MongoDB client connection.
    
    Returns:
        MongoClient: MongoDB client instance
    """
    try:
        client = MongoClient(config.MONGO_URI)
        # Ping the server to verify connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client
    except ConnectionFailure:
        logger.error("Failed to connect to MongoDB")
        raise ConnectionFailure("Could not connect to MongoDB. Please check your connection string.")

def get_database() -> Database:
    """
    Get the MongoDB database instance.
    
    Returns:
        Database: MongoDB database instance
    """
    client = get_mongo_client()
    return client[config.MONGO_DB_NAME]

def get_collection(collection_name: str) -> Collection:
    """
    Get a MongoDB collection by name.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        Collection: MongoDB collection instance
    """
    db = get_database()
    return db[collection_name]

def find_documents(
    collection_name: str, 
    query: Dict = None, 
    projection: Dict = None,
    sort: List = None,
    limit: int = 0
) -> List[Dict]:
    """
    Find documents in a MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        query: MongoDB query filter
        projection: Fields to include/exclude
        sort: Sort specification
        limit: Maximum number of documents to return
        
    Returns:
        List[Dict]: List of documents matching the query
    """
    try:
        collection = get_collection(collection_name)
        cursor = collection.find(query or {}, projection or {})
        
        if sort:
            cursor = cursor.sort(sort)
        
        if limit > 0:
            cursor = cursor.limit(limit)
            
        return list(cursor)
    except OperationFailure as e:
        logger.error(f"Database operation failed: {e}")
        raise

def find_one_document(
    collection_name: str, 
    query: Dict = None, 
    projection: Dict = None
) -> Optional[Dict]:
    """
    Find a single document in a MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        query: MongoDB query filter
        projection: Fields to include/exclude
        
    Returns:
        Dict or None: The document if found, None otherwise
    """
    try:
        collection = get_collection(collection_name)
        return collection.find_one(query or {}, projection or {})
    except OperationFailure as e:
        logger.error(f"Database operation failed: {e}")
        raise

def aggregate(collection_name: str, pipeline: List[Dict]) -> List[Dict]:
    """
    Perform an aggregation on a MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        pipeline: Aggregation pipeline
        
    Returns:
        List[Dict]: Aggregation results
    """
    try:
        collection = get_collection(collection_name)
        return list(collection.aggregate(pipeline))
    except OperationFailure as e:
        logger.error(f"Aggregation operation failed: {e}")
        raise

def collection_exists(collection_name: str) -> bool:
    """
    Check if a collection exists in the database.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        bool: True if collection exists, False otherwise
    """
    db = get_database()
    return collection_name in db.list_collection_names()

def get_distinct_values(collection_name: str, field: str) -> List[Any]:
    """
    Get distinct values for a field in a collection.
    
    Args:
        collection_name: Name of the collection
        field: Field name
        
    Returns:
        List: List of distinct values
    """
    try:
        collection = get_collection(collection_name)
        return collection.distinct(field)
    except OperationFailure as e:
        logger.error(f"Operation failed: {e}")
        raise