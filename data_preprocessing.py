import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.transactions = None
        self.basket_data = None
    
    def load_data(self, file_path):
        """Load transaction data"""
        try:
            # Try different separators
            self.transactions = pd.read_csv(file_path, header=None, names=['items'])
            print(f"✅ Data loaded: {len(self.transactions)} transactions")
            return self.transactions
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def create_sample_data(self):
        """Create sample grocery data if no file available"""
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
            'milk,cheese,yogurt,eggs'
        ]
        
        self.transactions = pd.DataFrame(sample_transactions, columns=['items'])
        print("✅ Sample data created: 15 transactions")
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
