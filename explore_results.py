# 🔍 Explore Your Market Basket Analysis Results
# Run these commands to see what you've discovered!

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and view your association rules
print("🏆 YOUR ASSOCIATION RULES:")
print("=" * 50)
rules = pd.read_csv('results/association_rules.csv')
print("Top 10 Rules by Lift:")
print(rules.head(10).to_string(index=False))

# 2. Load and view item frequencies  
print("\n📊 ITEM POPULARITY:")
print("=" * 30)
items = pd.read_csv('results/item_frequencies.csv')
print("Most Popular Items:")
print(items.head(8).to_string(index=False))

# 3. Analyze rule strength
print("\n📈 RULE STRENGTH ANALYSIS:")
print("=" * 40)
print(f"Total Rules Found: {len(rules)}")
print(f"Average Confidence: {rules['confidence'].mean():.3f}")
print(f"Average Lift: {rules['lift'].mean():.3f}")
print(f"Strong Rules (Lift > 2): {len(rules[rules['lift'] > 2])}")

# 4. Best Cross-Selling Opportunities
print("\n💰 BEST CROSS-SELLING OPPORTUNITIES:")
print("=" * 50)
high_confidence = rules[rules['confidence'] > 0.7].sort_values('lift', ascending=False)
print("High Confidence Rules (>70%):")
for idx, row in high_confidence.head(5).iterrows():
    print(f"• {row['rule']} (Confidence: {row['confidence']:.1%}, Lift: {row['lift']:.2f})")

# 5. Product Recommendation Function
def get_cart_recommendations(cart_items):
    """Get recommendations for a shopping cart"""
    print(f"\n🛒 CART: {cart_items}")
    print("-" * 30)
    
    recommendations = []
    cart_lower = [item.lower() for item in cart_items]
    
    for _, rule in rules.iterrows():
        rule_parts = rule['rule'].split(' → ')
        antecedent = rule_parts[0].strip()
        consequent = rule_parts[1].strip()
        
        # Check if antecedent items are in cart
        antecedent_items = [item.strip() for item in antecedent.split(',')]
        if all(item in cart_lower for item in antecedent_items):
            consequent_items = [item.strip() for item in consequent.split(',')]
            for item in consequent_items:
                if item not in cart_lower:
                    recommendations.append({
                        'item': item.title(),
                        'confidence': rule['confidence'],
                        'lift': rule['lift'],
                        'rule': rule['rule']
                    })
    
    # Sort by confidence and show top 3
    recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)[:3]
    
    if recommendations:
        print("💡 RECOMMENDED ITEMS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['item']} (Confidence: {rec['confidence']:.1%})")
            print(f"   Because: {rec['rule']}")
    else:
        print("No specific recommendations found for this cart")
    
    return recommendations

# 6. Test Different Shopping Carts
test_carts = [
    ['bread', 'butter'],
    ['milk', 'eggs'], 
    ['cheese', 'yogurt'],
    ['bread', 'jam'],
    ['eggs', 'butter']
]

print("\n🎯 TESTING RECOMMENDATION SYSTEM:")
print("=" * 50)

for cart in test_carts:
    get_cart_recommendations(cart)

# 7. Visualize Top Items
print("\n📊 CREATING VISUALIZATION...")
plt.figure(figsize=(10, 6))
top_items = items.head(8)
bars = plt.bar(range(len(top_items)), top_items['Frequency'], color='skyblue', edgecolor='navy')
plt.title('Most Popular Items in Your Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Items')
plt.ylabel('Frequency')
plt.xticks(range(len(top_items)), top_items['Item'], rotation=45)

# Add value labels on bars
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             str(top_items.iloc[i]['Frequency']), 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# 8. Business Insights Summary
print("\n📋 BUSINESS INSIGHTS SUMMARY:")
print("=" * 50)

# Find most confident rules
most_confident = rules.nlargest(3, 'confidence')
print("🎯 Most Reliable Rules:")
for _, rule in most_confident.iterrows():
    print(f"• {rule['rule']} ({rule['confidence']:.1%} confidence)")

# Find highest lift rules  
highest_lift = rules.nlargest(3, 'lift')
print(f"\n🚀 Strongest Associations (Highest Lift):")
for _, rule in highest_lift.iterrows():
    print(f"• {rule['rule']} (Lift: {rule['lift']:.2f})")

print(f"\n💡 KEY TAKEAWAYS:")
print(f"1. You have {len(rules)} actionable association rules")
print(f"2. Average rule confidence is {rules['confidence'].mean():.1%}")
print(f"3. {len(rules[rules['confidence'] > 0.8])} rules have >80% confidence")
print(f"4. Ready for production recommendation system!")

print(f"\n📁 All results saved in 'results/' folder")
print("🎉 Your Market Basket Analysis is complete and ready to use!")
