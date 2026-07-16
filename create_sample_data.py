#!/usr/bin/env python3
"""
Create sample grocery transaction data
"""

import pandas as pd
import numpy as np
import random
import os

def create_grocery_data(n_transactions=1000):
    """Create realistic grocery transaction data"""
    
    # Define product categories and items
    products = {
        'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream'],
        'bread': ['bread', 'bagel', 'croissant', 'muffin'],
        'meat': ['chicken', 'beef', 'pork', 'fish', 'turkey'],
        'produce': ['apple', 'banana', 'orange', 'carrot', 'lettuce', 'tomato'],
        'pantry': ['rice', 'pasta', 'cereal', 'oil', 'flour', 'sugar'],
        'beverages': ['coffee', 'tea', 'juice', 'soda', 'water'],
        'snacks': ['chips', 'cookies', 'crackers', 'nuts'],
        'frozen': ['ice_cream', 'frozen_pizza', 'frozen_vegetables'],
        'condiments': ['ketchup', 'mustard', 'mayo', 'jam', 'honey']
    }
    
    # Product associations (items often bought together)
    associations = [
        ['bread', 'butter', 'jam'],
        ['milk', 'cereal', 'banana'],
        ['pasta', 'tomato', 'cheese'],
        ['chicken', 'rice', 'vegetables'],
        ['coffee', 'milk', 'sugar'],
        ['bread', 'cheese', 'ham'],
        ['yogurt', 'honey', 'apple'],
        ['chips', 'soda', 'cookies'],
        ['ice_cream', 'cookies', 'milk']
    ]
    
    transactions = []
    
    for _ in range(n_transactions):
        transaction_items = []
        
        # Random transaction size (1-8 items)
        n_items = np.random.choice(range(1, 9), p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05])
        
        # 30% chance of using an association pattern
        if random.random() < 0.3 and random.choice(associations):
            base_items = random.choice(associations)
            transaction_items.extend(random.sample(base_items, min(len(base_items), n_items)))
        
        # Fill remaining slots with random items
        all_items = [item for category in products.values() for item in category]
        remaining_slots = max(0, n_items - len(transaction_items))
        
        if remaining_slots > 0:
            available_items = [item for item in all_items if item not in transaction_items]
            additional_items = random.sample(available_items, min(remaining_slots, len(available_items)))
            transaction_items.extend(additional_items)
        
        # Create transaction string
        transaction_str = ','.join(transaction_items)
        transactions.append(transaction_str)
    
    return transactions

def main():
    """Create and save sample data"""
    print("🛒 Creating sample grocery transaction data...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate transactions
    transactions = create_grocery_data(1000)
    
    # Save to CSV
    df = pd.DataFrame(transactions, columns=['items'])
    df.to_csv('data/groceries.csv', index=False, header=False)
    
    print(f"✅ Created {len(transactions)} sample transactions")
    print(f"💾 Saved to data/groceries.csv")
    
    # Show sample
    print(f"\n📋 Sample transactions:")
    for i in range(5):
        print(f"  {i+1}. {transactions[i]}")

if __name__ == "__main__":
    main()
