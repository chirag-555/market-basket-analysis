#!/usr/bin/env python3
"""
🚀 Real-time Market Basket Recommendation API
Production-ready FastAPI service with live recommendations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import uvicorn
from datetime import datetime, timedelta
import json
import asyncio
from collections import defaultdict, Counter
import logging
from pathlib import Path
# redis is optional — if not installed, falls back to in-memory caching
try:
    import redis as redis_lib
    REDIS_AVAILABLE_LIB = True
except ImportError:
    REDIS_AVAILABLE_LIB = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class CartItem(BaseModel):
    item_name: str
    quantity: int = 1
    price: Optional[float] = None

class RecommendationRequest(BaseModel):
    customer_id: Optional[str] = None
    cart_items: List[str]
    max_recommendations: int = 5
    min_confidence: float = 0.3
    include_explanations: bool = True

class Recommendation(BaseModel):
    item: str
    confidence: float
    lift: float
    support: float
    explanation: str
    estimated_price: Optional[float] = None
    category: Optional[str] = None

class RecommendationResponse(BaseModel):
    customer_id: Optional[str]
    cart_items: List[str]
    recommendations: List[Recommendation]
    total_rules_applied: int
    processing_time_ms: float
    timestamp: datetime

class TransactionLog(BaseModel):
    transaction_id: str
    customer_id: Optional[str]
    items: List[str]
    timestamp: datetime
    total_amount: Optional[float]

# Real-time Recommendation Engine
class RealTimeRecommendationEngine:
    def __init__(self):
        self.rules = None
        self.item_prices = {}
        self.item_categories = {}
        self.customer_profiles = {}
        self.transaction_log = []
        self.rule_performance = defaultdict(lambda: {'used': 0, 'success': 0})
        
        # Try to connect to Redis for caching (optional)
        try:
            if REDIS_AVAILABLE_LIB:
                self.redis_client = redis_lib.Redis(host='localhost', port=6379, decode_responses=True)
                self.redis_client.ping()  # test connection
                self.redis_available = True
                logger.info("✅ Redis connected for caching")
            else:
                raise ImportError("redis not installed")
        except Exception:
            self.redis_client = None
            self.redis_available = False
            logger.info("⚠️  Redis not available, using in-memory caching")
        
        self.load_models()
    
    def load_models(self):
        """Load association rules and supporting data"""
        try:
            # Load association rules
            if Path('results/association_rules.csv').exists():
                self.rules = pd.read_csv('results/association_rules.csv')
                logger.info(f"✅ Loaded {len(self.rules)} association rules")
            else:
                logger.warning("⚠️  No rules file found. Creating sample rules...")
                self.create_sample_rules()
            
            # Load item metadata
            self.load_item_metadata()
            
            # Load customer profiles if available
            self.load_customer_profiles()
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            self.create_sample_rules()
    
    def create_sample_rules(self):
        """Create sample association rules for demo"""
        sample_rules = [
            {'rule': 'bread → butter', 'support': 0.4, 'confidence': 0.8, 'lift': 2.1},
            {'rule': 'milk → cereal', 'support': 0.3, 'confidence': 0.7, 'lift': 1.9},
            {'rule': 'bread, butter → jam', 'support': 0.25, 'confidence': 0.75, 'lift': 2.3},
            {'rule': 'eggs → bacon', 'support': 0.2, 'confidence': 0.6, 'lift': 1.8},
            {'rule': 'coffee → sugar', 'support': 0.35, 'confidence': 0.85, 'lift': 2.5},
            {'rule': 'pasta → tomato_sauce', 'support': 0.3, 'confidence': 0.9, 'lift': 2.7},
            {'rule': 'chips → soda', 'support': 0.25, 'confidence': 0.65, 'lift': 1.7},
            {'rule': 'yogurt → granola', 'support': 0.2, 'confidence': 0.7, 'lift': 2.0}
        ]
        
        self.rules = pd.DataFrame(sample_rules)
        logger.info(f"✅ Created {len(sample_rules)} sample rules")
    
    def load_item_metadata(self):
        """Load item prices and categories"""
        # Sample item metadata
        self.item_prices = {
            'bread': 2.99, 'butter': 3.49, 'jam': 4.99, 'milk': 3.29,
            'cereal': 4.99, 'eggs': 2.99, 'bacon': 5.99, 'coffee': 8.99,
            'sugar': 2.49, 'pasta': 1.99, 'tomato_sauce': 2.29,
            'chips': 3.49, 'soda': 1.99, 'yogurt': 1.49, 'granola': 4.99
        }
        
        self.item_categories = {
            'bread': 'bakery', 'butter': 'dairy', 'jam': 'condiments',
            'milk': 'dairy', 'cereal': 'breakfast', 'eggs': 'dairy',
            'bacon': 'meat', 'coffee': 'beverages', 'sugar': 'pantry',
            'pasta': 'pantry', 'tomato_sauce': 'condiments', 
            'chips': 'snacks', 'soda': 'beverages', 'yogurt': 'dairy',
            'granola': 'breakfast'
        }
    
    def load_customer_profiles(self):
        """Load customer purchase history and preferences"""
        if Path('results/customer_segments.csv').exists():
            try:
                customer_data = pd.read_csv('results/customer_segments.csv')
                for _, row in customer_data.iterrows():
                    self.customer_profiles[row['customer_id']] = {
                        'segment': row.get('segment', 'unknown'),
                        'frequency': row.get('frequency', 1),
                        'monetary': row.get('monetary', 0),
                        'preferences': []
                    }
                logger.info(f"✅ Loaded {len(self.customer_profiles)} customer profiles")
            except Exception as e:
                logger.warning(f"⚠️  Error loading customer profiles: {e}")
    
    async def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """Get real-time recommendations for a cart"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"rec:{hash(str(sorted(request.cart_items)))}"
            if self.redis_available:
                cached = await self.get_from_cache(cache_key)
                if cached:
                    logger.info("📦 Returning cached recommendations")
                    return cached
            
            recommendations = []
            rules_applied = 0
            
            # Normalize cart items
            cart_items_lower = [item.lower().strip() for item in request.cart_items]
            
            # Apply association rules
            for _, rule in self.rules.iterrows():
                try:
                    # Parse rule
                    rule_parts = rule['rule'].split(' → ')
                    if len(rule_parts) != 2:
                        continue
                    
                    antecedent = rule_parts[0].strip()
                    consequent = rule_parts[1].strip()
                    
                    # Handle multiple antecedents
                    antecedent_items = [item.strip() for item in antecedent.split(',')]
                    consequent_items = [item.strip() for item in consequent.split(',')]
                    
                    # Check if all antecedents are in cart
                    if all(ant_item in cart_items_lower for ant_item in antecedent_items):
                        for cons_item in consequent_items:
                            if cons_item not in cart_items_lower and rule['confidence'] >= request.min_confidence:
                                
                                # Create recommendation
                                recommendation = Recommendation(
                                    item=cons_item.replace('_', ' ').title(),
                                    confidence=float(rule['confidence']),
                                    lift=float(rule['lift']),
                                    support=float(rule['support']),
                                    explanation=f"Customers who buy {', '.join(antecedent_items)} often also buy this",
                                    estimated_price=self.item_prices.get(cons_item),
                                    category=self.item_categories.get(cons_item)
                                )
                                
                                recommendations.append(recommendation)
                                rules_applied += 1
                                
                                # Track rule usage
                                self.rule_performance[rule['rule']]['used'] += 1
                
                except Exception as e:
                    logger.warning(f"Error processing rule {rule['rule']}: {e}")
                    continue
            
            # Sort by confidence * lift and remove duplicates
            seen_items = set()
            unique_recommendations = []
            
            for rec in sorted(recommendations, 
                            key=lambda x: x.confidence * x.lift, 
                            reverse=True):
                if rec.item not in seen_items:
                    seen_items.add(rec.item)
                    unique_recommendations.append(rec)
            
            # Limit to max recommendations
            final_recommendations = unique_recommendations[:request.max_recommendations]
            
            # Add customer-specific adjustments
            if request.customer_id and request.customer_id in self.customer_profiles:
                final_recommendations = self.adjust_for_customer(
                    final_recommendations, request.customer_id
                )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create response
            response = RecommendationResponse(
                customer_id=request.customer_id,
                cart_items=request.cart_items,
                recommendations=final_recommendations,
                total_rules_applied=rules_applied,
                processing_time_ms=processing_time,
                timestamp=datetime.now()
            )
            
            # Cache the response
            if self.redis_available:
                await self.cache_response(cache_key, response)
            
            logger.info(f"✅ Generated {len(final_recommendations)} recommendations in {processing_time:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def adjust_for_customer(self, recommendations: List[Recommendation], customer_id: str) -> List[Recommendation]:
        """Adjust recommendations based on customer profile"""
        try:
            profile = self.customer_profiles.get(customer_id, {})
            segment = profile.get('segment', 'unknown')
            
            # Boost recommendations based on customer segment
            for rec in recommendations:
                category = rec.category
                
                if segment == 'health_conscious' and category in ['dairy', 'produce']:
                    rec.confidence = min(1.0, rec.confidence * 1.2)
                elif segment == 'budget_conscious' and rec.estimated_price and rec.estimated_price < 3.0:
                    rec.confidence = min(1.0, rec.confidence * 1.1)
                elif segment == 'convenience_shopper' and category in ['frozen', 'snacks']:
                    rec.confidence = min(1.0, rec.confidence * 1.15)
            
            return sorted(recommendations, key=lambda x: x.confidence * x.lift, reverse=True)
        
        except Exception as e:
            logger.warning(f"Error adjusting for customer: {e}")
            return recommendations
    
    async def get_from_cache(self, key: str):
        """Get recommendation from cache"""
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    return RecommendationResponse.model_validate_json(cached_data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
    
    async def cache_response(self, key: str, response: RecommendationResponse):
        """Cache recommendation response"""
        try:
            if self.redis_client:
                self.redis_client.setex(
                    key, 
                    300,  # 5 minutes TTL
                    response.json()
                )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def log_transaction(self, transaction: TransactionLog):
        """Log transaction for learning"""
        self.transaction_log.append(transaction.dict())
        
        # Keep only recent transactions in memory
        if len(self.transaction_log) > 1000:
            self.transaction_log = self.transaction_log[-500:]
        
        logger.info(f"📝 Logged transaction {transaction.transaction_id}")
    
    def get_analytics(self):
        """Get recommendation engine analytics"""
        return {
            'total_rules': len(self.rules) if self.rules is not None else 0,
            'cached_customers': len(self.customer_profiles),
            'transactions_logged': len(self.transaction_log),
            'rule_performance': dict(self.rule_performance),
            'top_recommended_items': self.get_top_recommended_items()
        }
    
    def get_top_recommended_items(self):
        """Get most frequently recommended items"""
        item_counts = Counter()
        for perf in self.rule_performance.values():
            item_counts.update({'item': perf['used']})
        return dict(item_counts.most_common(10))

# Initialize the recommendation engine
recommendation_engine = RealTimeRecommendationEngine()

# FastAPI app setup
app = FastAPI(
    title="Market Basket Recommendation API",
    description="Real-time product recommendations using association rules",
    version="1.0.0"
)

# CORS middleware for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "🛒 Market Basket Recommendation API",
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get product recommendations for a shopping cart"""
    try:
        return await recommendation_engine.get_recommendations(request)
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log-transaction")
async def log_transaction(transaction: TransactionLog, background_tasks: BackgroundTasks):
    """Log a completed transaction for learning"""
    background_tasks.add_task(recommendation_engine.log_transaction, transaction)
    return {"message": "Transaction logged successfully", "transaction_id": transaction.transaction_id}

@app.get("/analytics")
async def get_analytics():
    """Get recommendation engine analytics and performance metrics"""
    return recommendation_engine.get_analytics()

@app.get("/rules")
async def get_rules(limit: int = 20):
    """Get association rules used by the engine"""
    if recommendation_engine.rules is not None:
        rules_data = recommendation_engine.rules.head(limit).to_dict('records')
        return {
            "total_rules": len(recommendation_engine.rules),
            "rules": rules_data
        }
    return {"total_rules": 0, "rules": []}

@app.get("/customer/{customer_id}")
async def get_customer_profile(customer_id: str):
    """Get customer profile and purchase history"""
    profile = recommendation_engine.customer_profiles.get(customer_id, {})
    if not profile:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    return {
        "customer_id": customer_id,
        "profile": profile,
        "recommendation_count": sum(
            1 for perf in recommendation_engine.rule_performance.values() 
            if perf['used'] > 0
        )
    }

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining with new transaction data"""
    background_tasks.add_task(retrain_recommendation_model)
    return {"message": "Model retraining started", "timestamp": datetime.now()}

def retrain_recommendation_model():
    """Background task to retrain the model"""
    try:
        logger.info("🔄 Starting model retraining...")
        # In production, this would retrain with new transaction data
        recommendation_engine.load_models()
        logger.info("✅ Model retraining completed")
    except Exception as e:
        logger.error(f"❌ Retraining error: {e}")

# Development server
if __name__ == "__main__":
    print("🚀 Starting Market Basket Recommendation API")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔗 API Base URL: http://localhost:8000")
    
    uvicorn.run(
        "realtime_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
      )
