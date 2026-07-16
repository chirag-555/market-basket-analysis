import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

class MarketBasketAnalyzer:
    def __init__(self):
        self.frequent_itemsets = None
        self.rules = None
        self.basket_data = None
    
    def fit(self, basket_data, min_support=0.1):
        """Fit the model using Apriori algorithm"""
        self.basket_data = basket_data
        
        print(f"🔍 Finding frequent itemsets with min_support={min_support}")
        
        # Apply Apriori algorithm
        self.frequent_itemsets = apriori(basket_data, 
                                       min_support=min_support, 
                                       use_colnames=True)
        
        if len(self.frequent_itemsets) == 0:
            print("❌ No frequent itemsets found. Try lowering min_support.")
            return None
        
        print(f"✅ Found {len(self.frequent_itemsets)} frequent itemsets")
        return self.frequent_itemsets
    
    def generate_rules(self, metric='lift', min_threshold=1.0):
        """Generate association rules"""
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            print("❌ No frequent itemsets available")
            return None
        
        try:
            self.rules = association_rules(self.frequent_itemsets, 
                                         metric=metric,
                                         min_threshold=min_threshold)
            
            if len(self.rules) == 0:
                print("❌ No rules found. Try lowering min_threshold.")
                return None
            
            # Add readable rule format
            self.rules['antecedents_str'] = self.rules['antecedents'].apply(
                lambda x: ', '.join(list(x))
            )
            self.rules['consequents_str'] = self.rules['consequents'].apply(
                lambda x: ', '.join(list(x))
            )
            self.rules['rule'] = self.rules['antecedents_str'] + ' → ' + self.rules['consequents_str']
            
            print(f"✅ Generated {len(self.rules)} association rules")
            return self.rules
        
        except Exception as e:
            print(f"❌ Error generating rules: {e}")
            return None
    
    def get_top_rules(self, n=10, sort_by='lift'):
        """Get top n rules sorted by specified metric"""
        if self.rules is None or len(self.rules) == 0:
            print("❌ No rules available")
            return None
        
        top_rules = self.rules.nlargest(n, sort_by)[
            ['rule', 'support', 'confidence', 'lift', 'conviction']
        ].round(3)
        
        return top_rules
    
    def rule_metrics_summary(self):
        """Get summary statistics of rule metrics"""
        if self.rules is None:
            print("❌ No rules available")
            return None
        
        summary = self.rules[['support', 'confidence', 'lift', 'conviction']].describe()
        return summary
    
    def get_recommendations(self, cart_items, top_n=3):
        """Get product recommendations based on current cart"""
        if self.rules is None:
            print("❌ No rules available")
            return []
        
        cart_items = [item.lower().strip() for item in cart_items]
        recommendations = []
        
        for _, rule in self.rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            # Check if cart contains all antecedents
            if all(item in cart_items for item in antecedents):
                for consequent in consequents:
                    if consequent not in cart_items:
                        recommendations.append({
                            'item': consequent,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'rule': rule['rule']
                        })
        
        # Sort by confidence and return top N
        recommendations = sorted(recommendations, 
                               key=lambda x: x['confidence'], 
                               reverse=True)
        
        return recommendations[:top_n]
