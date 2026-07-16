#!/usr/bin/env python3
"""
🚀 Advanced Market Basket Analysis Features
Enhanced version with powerful business applications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from collections import defaultdict
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedMarketBasketAnalyzer:
    def __init__(self):
        self.rules = None
        self.transactions = None
        self.customer_segments = None
        self.seasonal_patterns = None
    
    def load_rules(self, rules_path='results/association_rules.csv'):
        """Load existing association rules"""
        try:
            self.rules = pd.read_csv(rules_path)
            print(f"✅ Loaded {len(self.rules)} association rules")
            return self.rules
        except:
            print("❌ Could not load rules. Run basic analysis first.")
            return None
    
    def create_enhanced_transactions(self, n_customers=100, n_transactions=1000):
        """Create realistic transaction data with customer info and timestamps"""
        
        # Product categories with seasonal preferences
        products = {
            'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream'],
            'bakery': ['bread', 'croissant', 'muffin', 'bagel'],
            'meat': ['chicken', 'beef', 'fish', 'turkey'],
            'produce': ['apple', 'banana', 'orange', 'lettuce', 'tomato'],
            'beverages': ['coffee', 'tea', 'juice', 'soda'],
            'snacks': ['chips', 'cookies', 'crackers', 'nuts'],
            'frozen': ['ice_cream', 'frozen_pizza', 'frozen_vegetables'],
            'pantry': ['rice', 'pasta', 'oil', 'flour', 'cereal']
        }
        
        # Customer personas with different preferences
        customer_personas = {
            'health_conscious': {
                'preferences': ['produce', 'dairy', 'meat'],
                'avg_basket_size': 6,
                'frequency': 'high'
            },
            'convenience_shopper': {
                'preferences': ['frozen', 'snacks', 'beverages'],
                'avg_basket_size': 4,
                'frequency': 'medium'
            },
            'family_shopper': {
                'preferences': ['dairy', 'bakery', 'meat', 'produce'],
                'avg_basket_size': 8,
                'frequency': 'high'
            },
            'budget_conscious': {
                'preferences': ['pantry', 'bakery', 'dairy'],
                'avg_basket_size': 5,
                'frequency': 'low'
            }
        }
        
        transactions = []
        customers = {}
        
        # Create customers
        for i in range(n_customers):
            persona = random.choice(list(customer_personas.keys()))
            customers[f'CUST_{i:04d}'] = {
                'persona': persona,
                'preferences': customer_personas[persona]['preferences'],
                'avg_basket_size': customer_personas[persona]['avg_basket_size'],
                'total_spent': 0,
                'transaction_count': 0
            }
        
        # Generate transactions over past 12 months
        start_date = datetime.now() - timedelta(days=365)
        
        for _ in range(n_transactions):
            customer_id = random.choice(list(customers.keys()))
            customer = customers[customer_id]
            
            # Random date in past year
            random_days = random.randint(0, 365)
            transaction_date = start_date + timedelta(days=random_days)
            
            # Seasonal adjustments
            month = transaction_date.month
            seasonal_boost = {}
            if month in [12, 1, 2]:  # Winter
                seasonal_boost = {'frozen': 1.3, 'beverages': 0.7}
            elif month in [6, 7, 8]:  # Summer
                seasonal_boost = {'produce': 1.3, 'frozen': 1.2, 'beverages': 1.4}
            
            # Generate basket
            basket_size = max(1, int(np.random.normal(customer['avg_basket_size'], 2)))
            basket = []
            
            # Prefer customer's favorite categories
            available_categories = list(products.keys())
            preferred_categories = customer['preferences'] + available_categories
            
            for _ in range(basket_size):
                category = random.choice(preferred_categories)
                
                # Apply seasonal boost
                if category in seasonal_boost:
                    if random.random() < seasonal_boost[category] - 1:
                        category = random.choice(list(seasonal_boost.keys()))
                
                item = random.choice(products[category])
                if item not in basket:
                    basket.append(item)
            
            # Simulate item prices and calculate total
            item_prices = {item: round(random.uniform(1.50, 15.99), 2) for item in basket}
            total_amount = sum(item_prices.values())
            
            transaction = {
                'transaction_id': f'TXN_{len(transactions):06d}',
                'customer_id': customer_id,
                'date': transaction_date.strftime('%Y-%m-%d'),
                'items': ','.join(basket),
                'basket_size': len(basket),
                'total_amount': round(total_amount, 2),
                'persona': customer['persona']
            }
            
            transactions.append(transaction)
            
            # Update customer stats
            customers[customer_id]['total_spent'] += total_amount
            customers[customer_id]['transaction_count'] += 1
        
        self.transactions = pd.DataFrame(transactions)
        self.customers = pd.DataFrame.from_dict(customers, orient='index')
        self.customers.index.name = 'customer_id'
        
        print(f"✅ Generated {len(transactions)} enhanced transactions for {n_customers} customers")
        return self.transactions
    
    def customer_segmentation_analysis(self):
        """Segment customers based on purchasing behavior"""
        if self.transactions is None:
            print("❌ No transaction data available")
            return None
        
        # Calculate RFM metrics (Recency, Frequency, Monetary)
        customer_metrics = []
        current_date = pd.to_datetime(self.transactions['date']).max()
        
        for customer_id in self.transactions['customer_id'].unique():
            customer_data = self.transactions[self.transactions['customer_id'] == customer_id]
            
            # Recency: Days since last purchase
            last_purchase = pd.to_datetime(customer_data['date']).max()
            recency = (current_date - last_purchase).days
            
            # Frequency: Number of transactions
            frequency = len(customer_data)
            
            # Monetary: Total spent
            monetary = customer_data['total_amount'].sum()
            
            # Additional metrics
            avg_basket_size = customer_data['basket_size'].mean()
            
            customer_metrics.append({
                'customer_id': customer_id,
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary,
                'avg_basket_size': avg_basket_size,
                'persona': customer_data['persona'].iloc[0]
            })
        
        rfm_data = pd.DataFrame(customer_metrics)
        
        # Normalize for clustering
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_data[['recency', 'frequency', 'monetary', 'avg_basket_size']])
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm_data['cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Label clusters
        cluster_labels = {
            0: 'Champions',
            1: 'Loyal Customers', 
            2: 'Potential Loyalists',
            3: 'At Risk'
        }
        
        rfm_data['segment'] = rfm_data['cluster'].map(cluster_labels)
        self.customer_segments = rfm_data
        
        print("✅ Customer segmentation complete")
        return rfm_data
    
    def seasonal_pattern_analysis(self):
        """Analyze seasonal purchasing patterns"""
        if self.transactions is None:
            print("❌ No transaction data available")
            return None
        
        # Add date components
        self.transactions['date'] = pd.to_datetime(self.transactions['date'])
        self.transactions['month'] = self.transactions['date'].dt.month
        self.transactions['quarter'] = self.transactions['date'].dt.quarter
        self.transactions['day_of_week'] = self.transactions['date'].dt.day_name()
        
        # Seasonal analysis by month
        monthly_patterns = []
        
        for month in range(1, 13):
            month_data = self.transactions[self.transactions['month'] == month]
            
            if len(month_data) > 0:
                # Get all items for this month
                all_items = []
                for items_str in month_data['items']:
                    all_items.extend(items_str.split(','))
                
                item_counts = pd.Series(all_items).value_counts()
                
                monthly_patterns.append({
                    'month': month,
                    'transactions': len(month_data),
                    'avg_basket_size': month_data['basket_size'].mean(),
                    'avg_amount': month_data['total_amount'].mean(),
                    'top_item': item_counts.index[0] if len(item_counts) > 0 else 'N/A',
                    'top_item_count': item_counts.iloc[0] if len(item_counts) > 0 else 0
                })
        
        self.seasonal_patterns = pd.DataFrame(monthly_patterns)
        
        print("✅ Seasonal analysis complete")
        return self.seasonal_patterns
    
    def advanced_recommendation_engine(self, customer_id, cart_items=None):
        """Advanced recommendation engine considering customer profile"""
        if self.rules is None:
            print("❌ No rules loaded")
            return []
        
        recommendations = []
        
        # Get customer profile
        if self.customer_segments is not None:
            customer_profile = self.customer_segments[
                self.customer_segments['customer_id'] == customer_id
            ]
            
            if len(customer_profile) > 0:
                customer_segment = customer_profile['segment'].iloc[0]
                print(f"🎯 Customer Segment: {customer_segment}")
        
        # If no cart items, recommend based on customer history
        if cart_items is None or len(cart_items) == 0:
            if self.transactions is not None:
                customer_history = self.transactions[
                    self.transactions['customer_id'] == customer_id
                ]
                
                if len(customer_history) > 0:
                    # Get frequently bought items by this customer
                    all_items = []
                    for items_str in customer_history['items']:
                        all_items.extend(items_str.split(','))
                    
                    item_counts = pd.Series(all_items).value_counts()
                    cart_items = item_counts.head(3).index.tolist()
                    print(f"📝 Using customer history: {cart_items}")
        
        if cart_items:
            # Apply association rules
            for _, rule in self.rules.iterrows():
                rule_parts = rule['rule'].split(' → ')
                antecedent = rule_parts[0].strip()
                consequent = rule_parts[1].strip()
                
                antecedent_items = [item.strip() for item in antecedent.split(',')]
                if all(item in [c.lower() for c in cart_items] for item in antecedent_items):
                    consequent_items = [item.strip() for item in consequent.split(',')]
                    
                    for item in consequent_items:
                        if item not in [c.lower() for c in cart_items]:
                            recommendations.append({
                                'item': item.title(),
                                'confidence': rule['confidence'],
                                'lift': rule['lift'],
                                'reason': f"Often bought with {', '.join(antecedent_items)}"
                            })
            
            # Sort by confidence and remove duplicates
            recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
            
            # Remove duplicates
            seen = set()
            unique_recs = []
            for rec in recommendations:
                if rec['item'] not in seen:
                    seen.add(rec['item'])
                    unique_recs.append(rec)
            
            return unique_recs[:5]
        
        return []
    
    def create_interactive_dashboard(self):
        """Create visual dashboard of insights"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🛒 Market Basket Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Customer Segments
        if self.customer_segments is not None:
            segment_counts = self.customer_segments['segment'].value_counts()
            axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Customer Segments')
        
        # 2. Seasonal Patterns
        if self.seasonal_patterns is not None:
            axes[0, 1].plot(self.seasonal_patterns['month'], self.seasonal_patterns['avg_amount'], 
                           marker='o', linewidth=2, markersize=6)
            axes[0, 1].set_title('Average Spend by Month')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Average Amount ($)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Transaction Volume by Day
        if self.transactions is not None:
            day_counts = self.transactions['day_of_week'].value_counts()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = day_counts.reindex(day_order, fill_value=0)
            
            bars = axes[1, 0].bar(range(len(day_counts)), day_counts.values, color='skyblue')
            axes[1, 0].set_title('Transactions by Day of Week')
            axes[1, 0].set_xticks(range(len(day_counts)))
            axes[1, 0].set_xticklabels([d[:3] for d in day_counts.index], rotation=45)
            axes[1, 0].set_ylabel('Number of Transactions')
        
        # 4. Rule Strength Distribution
        if self.rules is not None:
            axes[1, 1].hist(self.rules['lift'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, 1].axvline(self.rules['lift'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {self.rules["lift"].mean():.2f}')
            axes[1, 1].set_title('Distribution of Rule Lift Values')
            axes[1, 1].set_xlabel('Lift')
            axes[1, 1].set_ylabel('Number of Rules')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_business_report(self):
        """Generate comprehensive business insights report"""
        report = f"""
# 📊 Advanced Market Basket Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Executive Summary
"""
        
        if self.customer_segments is not None:
            total_customers = len(self.customer_segments)
            avg_customer_value = self.customer_segments['monetary'].mean()
            
            report += f"""
### Customer Insights
- **Total Customers Analyzed**: {total_customers:,}
- **Average Customer Value**: ${avg_customer_value:.2f}
- **Customer Segments**: {len(self.customer_segments['segment'].unique())}

#### Top Customer Segments:
"""
            for segment in self.customer_segments['segment'].value_counts().head(3).index:
                count = self.customer_segments[self.customer_segments['segment'] == segment].shape[0]
                avg_value = self.customer_segments[self.customer_segments['segment'] == segment]['monetary'].mean()
                report += f"- **{segment}**: {count} customers (${avg_value:.2f} avg value)\n"
        
        if self.seasonal_patterns is not None:
            peak_month = self.seasonal_patterns.loc[self.seasonal_patterns['avg_amount'].idxmax(), 'month']
            peak_amount = self.seasonal_patterns['avg_amount'].max()
            
            report += f"""
### Seasonal Insights
- **Peak Sales Month**: {peak_month} (${peak_amount:.2f} average)
- **Seasonal Variation**: {(self.seasonal_patterns['avg_amount'].max() - self.seasonal_patterns['avg_amount'].min()):.2f}
"""
        
        if self.rules is not None:
            strong_rules = len(self.rules[self.rules['lift'] > 1.5])
            avg_confidence = self.rules['confidence'].mean()
            
            report += f"""
### Association Rule Insights
- **Total Rules Generated**: {len(self.rules)}
- **Strong Associations (Lift > 1.5)**: {strong_rules}
- **Average Rule Confidence**: {avg_confidence:.1%}

#### Top 5 Business Rules:
"""
            top_rules = self.rules.nlargest(5, 'confidence')
            for _, rule in top_rules.iterrows():
                report += f"- {rule['rule']} (Confidence: {rule['confidence']:.1%})\n"
        
        report += """
## 💡 Strategic Recommendations

### 1. Cross-Selling Opportunities
- Implement rule-based product recommendations at checkout
- Bundle frequently associated products for promotional offers
- Train sales staff on key product associations

### 2. Customer Retention Strategies
- Target 'At Risk' customers with personalized offers
- Develop loyalty programs for 'Champions' segment
- Re-engage 'Potential Loyalists' with targeted campaigns

### 3. Inventory Management
- Stock complementary items together based on association rules
- Plan seasonal inventory based on monthly patterns
- Optimize product placement using basket analysis insights

### 4. Marketing Initiatives
- Create targeted campaigns for each customer segment
- Time promotions based on seasonal purchasing patterns
- Use association rules for email marketing recommendations

## 📈 Expected Business Impact
- **Revenue Increase**: 5-15% through cross-selling
- **Customer Retention**: 10-20% improvement through targeted strategies
- **Inventory Efficiency**: 15-25% reduction in overstock
- **Marketing ROI**: 20-40% improvement through personalization
"""
        
        return report

def main():
    """Demonstrate advanced features"""
    print("🚀 Advanced Market Basket Analysis Features")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = AdvancedMarketBasketAnalyzer()
    
    # Load existing rules
    print("\n📊 Loading Association Rules...")
    analyzer.load_rules()
    
    # Generate enhanced transaction data
    print("\n🏪 Creating Enhanced Transaction Dataset...")
    transactions = analyzer.create_enhanced_transactions(n_customers=50, n_transactions=500)
    
    # Customer segmentation
    print("\n👥 Performing Customer Segmentation...")
    segments = analyzer.customer_segmentation_analysis()
    
    if segments is not None:
        print("\n🎯 Customer Segments Found:")
        segment_summary = segments.groupby('segment').agg({
            'monetary': ['count', 'mean'],
            'frequency': 'mean',
            'recency': 'mean'
        }).round(2)
        print(segment_summary)
    
    # Se
