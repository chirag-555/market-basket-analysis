import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

class MarketBasketVisualizer:
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_item_frequency(self, freq_df, top_n=15):
        """Plot frequency of items"""
        plt.figure(figsize=(12, 6))
        
        # Top items
        top_items = freq_df.head(top_n)
        
        plt.subplot(1, 2, 1)
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
        
        # Distribution
        plt.subplot(1, 2, 2)
        plt.hist(freq_df['Frequency'], bins=10, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Item Frequencies')
        plt.xlabel('Frequency')
        plt.ylabel('Number of Items')
        
        plt.tight_layout()
        plt.show()
    
    def plot_support_confidence_scatter(self, rules):
        """Scatter plot of support vs confidence"""
        if rules is None or len(rules) == 0:
            print("❌ No rules to plot")
            return
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(rules['support'], rules['confidence'], 
                            c=rules['lift'], cmap='viridis', 
                            alpha=0.7, s=60)
        plt.colorbar(scatter, label='Lift')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Association Rules: Support vs Confidence (colored by Lift)')
        plt.grid(True, alpha=0.3)
        
        # Add annotations for high-lift rules
        high_lift_rules = rules[rules['lift'] > rules['lift'].quantile(0.8)]
        for _, rule in high_lift_rules.iterrows():
            plt.annotate(rule['rule'][:20] + '...', 
                        (rule['support'], rule['confidence']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.show()
    
    def plot_rule_metrics_heatmap(self, rules, top_n=15):
        """Heatmap of rule metrics"""
        if rules is None or len(rules) == 0:
            print("❌ No rules to plot")
            return
        
        # Get top rules
        top_rules = rules.nlargest(top_n, 'lift')
        
        # Create matrix of metrics
        metrics_data = top_rules[['support', 'confidence', 'lift']].values
        rule_labels = [rule[:30] + '...' if len(rule) > 30 else rule 
                      for rule in top_rules['rule']]
        
        plt.figure(figsize=(8, max(6, top_n * 0.4)))
        sns.heatmap(metrics_data, 
                   xticklabels=['Support', 'Confidence', 'Lift'],
                   yticklabels=rule_labels,
                   annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title(f'Top {top_n} Association Rules - Metrics Heatmap')
        plt.tight_layout()
        plt.show()
    
    def plot_network_graph(self, rules, top_n=10):
        """Network graph of association rules"""
        if rules is None or len(rules) == 0:
            print("❌ No rules to plot")
            return
        
        # Get top rules
        top_rules = rules.nlargest(top_n, 'lift')
        
        # Create network graph
        G = nx.DiGraph()
        
        for _, rule in top_rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            # Add nodes
            for ant in antecedents:
                G.add_node(ant, node_type='antecedent')
            for con in consequents:
                G.add_node(con, node_type='consequent')
            
            # Add edges
            for ant in antecedents:
                for con in consequents:
                    G.add_edge(ant, con, 
                             weight=rule['lift'],
                             confidence=rule['confidence'])
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        node_colors = ['lightblue' if G.nodes[node].get('node_type') == 'antecedent' 
                      else 'lightcoral' for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=1000, alpha=0.8)
        
        # Draw edges with varying thickness based on lift
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [3 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              alpha=0.6, edge_color='gray', arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title(f'Association Rules Network (Top {top_n} by Lift)')
        plt.legend(['Antecedents (Items Bought)', 'Consequents (Recommended Items)'])
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_lift_distribution(self, rules):
        """Distribution of lift values"""
        if rules is None or len(rules) == 0:
            print("❌ No rules to plot")
            return
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(rules['lift'], bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(rules['lift'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {rules["lift"].mean():.2f}')
        plt.axvline(1, color='orange', linestyle='--', 
                   label='Independence (Lift = 1)')
        plt.xlabel('Lift')
        plt.ylabel('Number of Rules')
        plt.title('Distribution of Lift Values')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.boxplot(rules['lift'])
        plt.ylabel('Lift')
        plt.title('Lift Values Box Plot')
        
        plt.tight_layout()
        plt.show()
