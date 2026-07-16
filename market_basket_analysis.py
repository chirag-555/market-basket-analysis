#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Market Basket Analysis - Main Script
Encoding issue resolved
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Try importing, if fails create inline classes
try:
    from data_preprocessing import DataPreprocessor
    from market_basket_analyzer import MarketBasketAnalyzer
    from visualizations import MarketBasketVisualizer
except ImportError:
    print("📦 Creating inline classes...")
    
    # Inline DataPreprocessor
    from collections import Counter
    
    class DataPreprocessor:
        def __init__(self):
            self.transactions = None
            self.basket_data = None
        
        def create_sample_data(self):
            """Create sample grocery data"""
            sample_transactions = [
                'bread,milk,eggs',
                'bread,butter,jam',
                'milk,eggs,cheese',
                'bread,milk,butter',
                'eggs,cheese,yogurt',
                'bread,jam,butter',
                'milk,cheese,yogurt',
                'bread,eggs,butter',
                'milk,jam,cheese',
                'eggs,yogurt,butter',
                'bread,milk,cheese,eggs',
                'butter,jam,bread,milk',
                'yogurt,cheese,milk',
                'eggs,bread,butter,jam',
                'milk,cheese,yogurt,eggs',
                'bread,butter,milk,jam',
                'cheese,eggs,yogurt',
                'bread,milk,jam',
                'butter,eggs,cheese',
                'milk,yogurt,bread'
            ]
            
            self.transactions = pd.DataFrame(sample_transactions, columns=['items'])
            print(f"✅ Sample data created: {len(sample_transactions)} transactions")
            return self.transactions
        
        def clean_data(self):
            """Clean and preprocess transaction data"""
            if self.transactions is None:
                print("❌ No data to clean")
                return None
            
            # Remove empty transactions
            self.transactions = self.transactions.dropna()
            self.transactions = self.transactions[self.transactions['items'].str.strip() != '']
            
            # Split items and clean
            self.transactions['items_list'] = self.transactions['items'].str.split(',')
            self.transactions['items_list'] = self.transactions['items_list'].apply(
                lambda x: [item.strip().lower() for item in x if item.strip()]
            )
            
            print(f"✅ Data cleaned: {len(self.transactions)} valid transactions")
            return self.transactions
        
        def convert_to_basket(self):
            """Convert to basket format for association rules"""
            if self.transactions is None:
                print("❌ No data to convert")
                return None
            
            # Get all unique items
            all_items = set()
            for items_list in self.transactions['items_list']:
                all_items.update(items_list)
            
            all_items = sorted(list(all_items))
            
            # Create basket matrix
            basket_data = []
            for items_list in self.transactions['items_list']:
                basket = [1 if item in items_list else 0 for item in all_items]
                basket_data.append(basket)
            
            self.basket_data = pd.DataFrame(basket_data, columns=all_items)
            print(f"✅ Basket format created: {self.basket_data.shape}")
            return self.basket_data
        
        def get_item_frequencies(self):
            """Get frequency of each item"""
            all_items = []
            for items_list in self.transactions['items_list']:
                all_items.extend(items_list)
            
            item_freq = Counter(all_items)
            freq_df = pd.DataFrame(list(item_freq.items()), 
                                  columns=['Item', 'Frequency'])
            freq_df = freq_df.sort_values('Frequency', ascending=False)
            
            return freq_df
    
    # Inline MarketBasketAnalyzer
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        MLXTEND_AVAILABLE = True
    except ImportError:
        print("⚠️  mlxtend not available. Installing...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mlxtend'])
            from mlxtend.frequent_patterns import apriori, association_rules
            MLXTEND_AVAILABLE = True
        except:
            MLXTEND_AVAILABLE = False
            print("❌ Could not install mlxtend. Using basic analysis.")
    
    class MarketBasketAnalyzer:
        def __init__(self):
            self.frequent_itemsets = None
            self.rules = None
            self.basket_data = None
        
        def fit(self, basket_data, min_support=0.1):
            """Fit the model using Apriori algorithm"""
            self.basket_data = basket_data
            
            if not MLXTEND_AVAILABLE:
                print("❌ mlxtend not available for Apriori")
                return None
            
            print(f"🔍 Finding frequent itemsets with min_support={min_support}")
            
            try:
                # Apply Apriori algorithm
                self.frequent_itemsets = apriori(basket_data, 
                                               min_support=min_support, 
                                               use_colnames=True)
                
                if len(self.frequent_itemsets) == 0:
                    print("❌ No frequent itemsets found. Trying lower support...")
                    self.frequent_itemsets = apriori(basket_data, 
                                                   min_support=0.05, 
                                                   use_colnames=True)
                
                print(f"✅ Found {len(self.frequent_itemsets)} frequent itemsets")
                return self.frequent_itemsets
            except Exception as e:
                print(f"❌ Error in Apriori: {e}")
                return None
        
        def generate_rules(self, metric='lift', min_threshold=1.0):
            """Generate association rules"""
            if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
                print("❌ No frequent itemsets available")
                return None
            
            if not MLXTEND_AVAILABLE:
                print("❌ mlxtend not available for rules")
                return None
            
            try:
                self.rules = association_rules(self.frequent_itemsets, 
                                             metric=metric,
                                             min_threshold=min_threshold)
                
                if len(self.rules) == 0:
                    print("❌ No rules found. Trying lower threshold...")
                    self.rules = association_rules(self.frequent_itemsets, 
                                                 metric=metric,
                                                 min_threshold=0.1)
                
                if len(self.rules) == 0:
                    print("❌ Still no rules found.")
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
                ['rule', 'support', 'confidence', 'lift']
            ].round(3)
            
            return top_rules
        
        def get_recommendations(self, cart_items, top_n=3):
            """Get product recommendations based on current cart"""
            if self.rules is None or len(self.rules) == 0:
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
    
    # Simple Visualizer
    import matplotlib.pyplot as plt
    
    class MarketBasketVisualizer:
        def __init__(self):
            pass
        
        def plot_item_frequency(self, freq_df, top_n=10):
            """Plot frequency of items"""
            try:
                plt.figure(figsize=(12, 5))
                
                # Top items
                top_items = freq_df.head(top_n)
                
                bars = plt.bar(range(len(top_items)), top_items['Frequency'])
                plt.title(f'Top {top_n} Most Frequent Items')
                plt.xlabel('Items')
                plt.ylabel('Frequency')
                plt.xticks(range(len(top_items)), top_items['Item'], rotation=45)
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(top_items.iloc[i]['Frequency']), 
                            ha='center', va='bottom')
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"❌ Visualization error: {e}")

def main():
    """Main analysis pipeline with encoding fixes"""
    print("🛒 Market Basket Analysis Pipeline (Fixed)")
    print("=" * 50)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    analyzer = MarketBasketAnalyzer()
    visualizer = MarketBasketVisualizer()
    
    # Step 1: Load and preprocess data
    print("\n📊 Step 1: Data Loading and Preprocessing")
    print("-" * 40)
    
    # Create sample data (avoids file encoding issues)
    transactions = preprocessor.create_sample_data()
    
    # Clean and convert data
    cleaned_data = preprocessor.clean_data()
    basket_data = preprocessor.convert_to_basket()
    
    if basket_data is None:
        print("❌ Failed to create basket data")
        return
    
    # Step 2: EDA
    print("\n🔍 Step 2: Exploratory Data Analysis")
    print("-" * 40)
    
    freq_df = preprocessor.get_item_frequencies()
    print(f"📈 Top 5 most frequent items:")
    print(freq_df.head())
    
    print(f"\n📊 Dataset statistics:")
    print(f"  • Total transactions: {len(transactions)}")
    print(f"  • Unique items: {len(freq_df)}")
    print(f"  • Average items per transaction: {freq_df['Frequency'].sum() / len(transactions):.2f}")
    
    # Visualizations
    print("\n📈 Generating visualizations...")
    try:
        visualizer.plot_item_frequency(freq_df, top_n=8)
    except Exception as e:
        print(f"⚠️  Visualization skipped: {e}")
    
    # Step 3: Association Rule Mining
    print("\n⚡ Step 3: Association Rule Mining")
    print("-" * 40)
    
    # Try different support thresholds
    support_thresholds = [0.15, 0.1, 0.05]
    rules = None
    
    for min_sup in support_thresholds:
        print(f"\n🔍 Trying min_support = {min_sup}")
        frequent_itemsets = analyzer.fit(basket_data, min_support=min_sup)
        
        if frequent_itemsets is not None and len(frequent_itemsets) > 0:
            rules = analyzer.generate_rules(metric='lift', min_threshold=1.0)
            if rules is not None and len(rules) > 0:
                break
    
    if rules is None or len(rules) == 0:
        print("❌ No association rules found.")
        print("💡 This is normal with small datasets. Try adding more transactions.")
        return
    
    # Step 4: Rule Analysis
    print("\n📊 Step 4: Rule Analysis and Evaluation")
    print("-" * 40)
    
    # Top rules
    print(f"\n🏆 Top 10 Association Rules (by Lift):")
    top_rules = analyzer.get_top_rules(n=10, sort_by='lift')
    if top_rules is not None:
        print(top_rules.to_string(index=False))
    
    # Step 5: Save Results
    print("\n💾 Step 5: Saving Results")
    print("-" * 40)
    
    try:
        # Save rules with proper encoding
        if rules is not None:
            rules_output = rules[['rule', 'support', 'confidence', 'lift']].round(4)
            rules_output.to_csv('results/association_rules.csv', 
                              index=False, encoding='utf-8')
            print("✅ Rules saved to results/association_rules.csv")
        
        # Save item frequencies
        freq_df.to_csv('results/item_frequencies.csv', 
                       index=False, encoding='utf-8')
        print("✅ Item frequencies saved to results/item_frequencies.csv")
        
    except Exception as e:
        print(f"⚠️  Save error: {e}")
    
    # Step 6: Recommendation Demo
    print("\n🎯 Step 6: Product Recommendation Demo")
    print("-" * 40)
    
    # Demo recommendations
    sample_carts = [
        ['bread', 'milk'],
        ['eggs', 'cheese'],
        ['butter', 'jam']
    ]
    
    for i, cart in enumerate(sample_carts, 1):
        print(f"\n🛒 Demo Cart {i}: {cart}")
        recommendations = analyzer.get_recommendations(cart, top_n=3)
        
        if recommendations:
            print("💡 Recommended items:")
            for j, rec in enumerate(recommendations, 1):
                print(f"  {j}. {rec['item']} (confidence: {rec['confidence']:.3f})")
        else:
            print("  No specific recommendations found for this cart.")
    
    # Step 7: Generate Report
    print("\n📋 Step 7: Generating Summary Report")
    print("-" * 40)
    
    try:
        report_content = f"""# Market Basket Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
- Total Transactions: {len(transactions)}
- Unique Items: {len(freq_df)}
- Average Items per Transaction: {freq_df['Frequency'].sum() / len(transactions):.2f}

## Top Items
{freq_df.head().to_string(index=False)}

## Analysis Results
- Association Rules Generated: {len(rules) if rules is not None else 0}
- Analysis completed successfully with sample data

## Business Insights
1. Most popular items identified for inventory planning
2. Cross-selling opportunities discovered through association rules
3. Recommendation system ready for implementation

## Next Steps
1. Replace sample data with real transaction data
2. Adjust support and confidence thresholds based on business needs
3. Implement recommendation system in production
"""
        
        # Save with proper encoding
        with open('results/analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("✅ Report saved to results/analysis_report.md")
        
    except Exception as e:
        print(f"⚠️  Report save error: {e}")
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📁 Check 'results/' folder for output files")
    
    if rules is not None and len(rules) > 0:
        print(f"🔗 Found {len(rules)} association rules")
        print("💡 Try the recommendation system with your own cart items!")
    else:
        print("💡 To get better results, increase your dataset size")

if __name__ == "__main__":
    main()
