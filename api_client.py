#!/usr/bin/env python3
"""
🧪 Market Basket API Client & Test Suite
Test the real-time recommendation API
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import asyncio
import aiohttp
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

console = Console()

class MarketBasketAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_recommendations(self, cart_items: List[str], customer_id: str = None, 
                          max_recommendations: int = 5) -> Dict[str, Any]:
        """Get recommendations for a shopping cart"""
        url = f"{self.base_url}/recommendations"
        
        payload = {
            "cart_items": cart_items,
            "max_recommendations": max_recommendations,
            "min_confidence": 0.3,
            "include_explanations": True
        }
        
        if customer_id:
            payload["customer_id"] = customer_id
        
        try:
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"❌ API Error: {e}", style="red")
            return {}
    
    def log_transaction(self, transaction_id: str, items: List[str], 
                       customer_id: str = None, total_amount: float = None):
        """Log a completed transaction"""
        url = f"{self.base_url}/log-transaction"
        
        payload = {
            "transaction_id": transaction_id,
            "items": items,
            "timestamp": datetime.now().isoformat(),
        }
        
        if customer_id:
            payload["customer_id"] = customer_id
        if total_amount:
            payload["total_amount"] = total_amount
        
        try:
            response = self.session.post(url, json=payload, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"❌ Transaction logging error: {e}", style="red")
            return {}
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get API analytics"""
        try:
            response = self.session.get(f"{self.base_url}/analytics", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"❌ Analytics error: {e}", style="red")
            return {}
    
    def get_rules(self, limit: int = 10) -> Dict[str, Any]:
        """Get association rules"""
        try:
            response = self.session.get(f"{self.base_url}/rules?limit={limit}", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"❌ Rules error: {e}", style="red")
            return {}
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.session.get(self.base_url, timeout=5)
            response.raise_for_status()
            return True
        except:
            return False

def display_recommendations(recommendations_data: Dict[str, Any]):
    """Display recommendations in a nice table"""
    if not recommendations_data or 'recommendations' not in recommendations_data:
        console.print("❌ No recommendations received", style="red")
        return
    
    recs = recommendations_data['recommendations']
    cart = recommendations_data.get('cart_items', [])
    processing_time = recommendations_data.get('processing_time_ms', 0)
    
    console.print(f"\n🛒 Cart: {', '.join(cart)}", style="bold blue")
    console.print(f"⚡ Processing Time: {processing_time:.1f}ms", style="dim")
    
    if not recs:
        console.print("💡 No recommendations found for this cart", style="yellow")
        return
    
    table = Table(title="💡 Product Recommendations")
    table.add_column("Item", style="cyan", no_wrap=True)
    table.add_column("Confidence", style="green")
    table.add_column("Lift", style="magenta")
    table.add_column("Price", style="yellow")
    table.add_column("Explanation", style="dim")
    
    for rec in recs:
        confidence = f"{rec['confidence']:.1%}"
        lift = f"{rec['lift']:.2f}"
        price = f"${rec['estimated_price']:.2f}" if rec['estimated_price'] else "N/A"
        explanation = rec['explanation'][:50] + "..." if len(rec['explanation']) > 50 else rec['explanation']
        
        table.add_row(rec['item'], confidence, lift, price, explanation)
    
    console.print(table)

def run_interactive_demo(client: MarketBasketAPIClient):
    """Interactive demo of the API"""
    console.print("🎯 Interactive Market Basket Recommendation Demo", style="bold green")
    console.print("Type 'exit' to quit, 'help' for commands\n")
    
    sample_carts = [
        ['bread', 'butter'],
        ['milk', 'cereal'], 
        ['pasta', 'tomato_sauce'],
        ['coffee', 'sugar'],
        ['chips', 'soda'],
        ['yogurt', 'granola']
    ]
    
    while True:
        try:
            console.print("\n" + "="*50)
            user_input = input("🛒 Enter cart items (comma-separated) or 'sample' for examples: ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'help':
                console.print("""
Available commands:
- Enter items: bread,milk,eggs
- 'sample': Try sample carts
- 'analytics': View API analytics  
- 'rules': View association rules
- 'exit': Quit demo
                """)
                continue
            elif user_input.lower() == 'sample':
                console.print("📋 Sample carts available:")
                for i, cart in enumerate(sample_carts, 1):
                    console.print(f"  {i}. {', '.join(cart)}")
                
                choice = input("Choose a cart number (1-6): ").strip()
                try:
                    cart_items = sample_carts[int(choice) - 1]
                except (ValueError, IndexError):
                    console.print("❌ Invalid choice", style="red")
                    continue
            elif user_input.lower() == 'analytics':
                analytics = client.get_analytics()
                if analytics:
                    console.print("\n📊 API Analytics:", style="bold")
                    for key, value in analytics.items():
                        console.print(f"  {key}: {value}")
                continue
            elif user_input.lower() == 'rules':
                rules_data = client.get_rules(limit=10)
                if rules_data and 'rules' in rules_data:
                    console.print(f"\n📋 Association Rules (showing {len(rules_data['rules'])} of {rules_data['total_rules']}):")
                    for rule in rules_data['rules']:
                        console.print(f"  • {rule['rule']} (confidence: {rule['confidence']:.1%})")
                continue
            else:
                # Parse user input
                cart_items = [item.strip().lower() for item in user_input.split(',') if item.strip()]
                
                if not cart_items:
                    console.print("❌ Please enter some items", style="red")
                    continue
            
            # Get recommendations
            console.print(f"🔍 Getting recommendations for: {cart_items}")
            
            start_time = time.time()
            recommendations = client.get_recommendations(cart_items, max_recommendations=5)
            request_time = (time.time() - start_time) * 1000
            
            if recommendations:
                display_recommendations(recommendations)
                console.print(f"🌐 Request Time: {request_time:.1f}ms", style="dim")
            else:
                console.print("❌ No response from API", style="red")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"❌ Error: {e}", style="red")
    
    console.print("\n👋 Thanks for trying the Market Basket API!", style="green")

def run_performance_test(client: MarketBasketAPIClient, num_requests: int = 50):
    """Performance test the API"""
    console.print(f"🚀 Running Performance Test ({num_requests} requests)", style="bold yellow")
    
    test_carts = [
        ['bread', 'butter'],
        ['milk', 'eggs', 'cheese'],
        ['pasta', 'tomato_sauce', 'cheese'],
        ['coffee', 'sugar', 'milk'],
        ['chips', 'soda', 'cookies'],
        ['yogurt', 'granola', 'honey'],
        ['chicken', 'rice', 'vegetables'],
        ['bread', 'jam', 'butter', 'milk']
    ]
    
    response_times = []
    successful_requests = 0
    
    with Progress() as progress:
        task = progress.add_task("Testing API...", total=num_requests)
        
        for i in range(num_requests):
            cart = test_carts[i % len(test_carts)]
            
            start_time = time.time()
            result = client.get_recommendations(cart, customer_id=f"test_customer_{i%10}")
            request_time = (time.time() - start_time) * 1000
            
            if result:
                response_times.append(request_time)
                successful_requests += 1
            
            progress.update(task, advance=1)
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.01)
    
    # Calculate statistics
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        success_rate = (successful_requests / num_requests) * 100
        
        # Results table
        table = Table(title="📊 Performance Test Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Requests", str(num_requests))
        table.add_row("Successful Requests", str(successful_requests))
        table.add_row("Success Rate", f"{success_rate:.1f}%")
        table.add_row("Average Response Time", f"{avg_time:.1f}ms")
        table.add_row("Min Response Time", f"{min_time:.1f}ms")
        table.add_row("Max Response Time", f"{max_time:.1f}ms")
        table.add_row("Requests/Second", f"{1000/avg_time:.1f}")
        
        console.print(table)
        
        # Performance assessment
        if avg_time < 100:
            console.print("🚀 Excellent performance!", style="green")
        elif avg_time < 500:
            console.print("✅ Good performance", style="yellow")
        else:
            console.print("⚠️  Consider performance optimization", style="red")
    
    else:
        console.print("❌ No successful requests", style="red")

def simulate_real_shopping_session(client: MarketBasketAPIClient):
    """Simulate a realistic shopping session"""
    console.print("🛍️  Simulating Real Shopping Session", style="bold cyan")
    
    # Shopping session scenario
    customer_id = "CUST_12345"
    session_items = []
    
    shopping_steps = [
        (['bread'], "Customer picks up bread"),
        (['bread', 'butter'], "Adds butter (recommended)"),
        (['bread', 'butter', 'jam'], "Adds jam (recommended)"),
        (['bread', 'butter', 'jam', 'milk'], "Adds milk for breakfast"),
        (['bread', 'butter', 'jam', 'milk', 'eggs'], "Adds eggs (recommended)"),
    ]
    
    total_recommendations_followed = 0
    
    for step, (cart, description) in enumerate(shopping_steps, 1):
        console.print(f"\n📍 Step {step}: {description}")
        console.print(f"🛒 Current cart: {', '.join(cart)}")
        
        # Get recommendations for current cart
        recommendations = client.get_recommendations(cart, customer_id=customer_id)
        
        if recommendations and recommendations.get('recommendations'):
            recs = recommendations['recommendations'][:3]  # Top 3
            
            console.print("💡 Real-time suggestions:")
            for i, rec in enumerate(recs, 1):
                console.print(f"  {i}. {rec['item']} ({rec['confidence']:.1%} confidence)")
            
            # Simulate customer following a recommendation
            if step < len(shopping_steps) - 1:
                next_step_items = shopping_steps[step][0]
                current_items = set(cart)
                new_items = set(next_step_items) - current_items
                
                for new_item in new_items:
                    for rec in recs:
                        if rec['item'].lower() == new_item.lower():
                            total_recommendations_followed += 1
                            console.print(f"✅ Customer followed recommendation: {new_item}", style="green")
                            break
        
        time.sleep(1)  # Simulate time between steps
    
    # Complete the transaction
    final_cart = shopping_steps[-1][0]
    transaction_id = f"TXN_{int(time.time())}"
    
    console.print(f"\n💳 Completing transaction: {transaction_id}")
    client.log_transaction(
        transaction_id=transaction_id,
        items=final_cart,
        customer_id=customer_id,
        total_amount=25.47
    )
    
    console.print(f"✅ Session complete! Customer followed {total_recommendations_followed} recommendations")

def main():
    """Main demo application"""
    console.print("🚀 Market Basket API Client & Test Suite", style="bold green")
    console.print("=" * 50)
    
    # Initialize client
    client = MarketBasketAPIClient()
    
    # Health check
    console.print("🔍 Checking API health...")
    if not client.health_check():
        console.print("❌ API is not accessible at http://localhost:8000", style="red")
        console.print("💡 Make sure to start the API server first:")
        console.print("   python realtime_api.py", style="yellow")
        return
    
    console.print("✅ API is healthy and ready!", style="green")
    
    while True:
        console.print("\n📋 Choose an option:")
        console.print("1. 🎯 Interactive Demo")
        console.print("2. 🚀 Performance Test") 
        console.print("3. 🛍️  Shopping Session Simulation")
        console.print("4. 📊 View API Analytics")
        console.print("5. 📋 View Association Rules")
        console.print("6. 🧪 Quick Test")
        console.print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        try:
            if choice == '1':
                run_interactive_demo(client)
            elif choice == '2':
                num_requests = input("Enter number of requests (default 50): ").strip()
                num_requests = int(num_requests) if num_requests else 50
                run_performance_test(client, num_requests)
            elif choice == '3':
                simulate_real_shopping_session(client)
            elif choice == '4':
                analytics = client.get_analytics()
                if analytics:
                    console.print("\n📊 API Analytics:", style="bold")
                    table = Table()
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")
                    
                    for key, value in analytics.items():
                        if isinstance(value, dict):
                            value = f"{len(value)} items"
                        table.add_row(str(key), str(value))
                    
                    console.print(table)
            elif choice == '5':
                rules_data = client.get_rules(limit=15)
                if rules_data and 'rules' in rules_data:
                    console.print(f"\n📋 Association Rules ({rules_data['total_rules']} total):")
                    
                    table = Table()
                    table.add_column("Rule", style="cyan")
                    table.add_column("Confidence", style="green")
                    table.add_column("Lift", style="magenta")
                    table.add_column("Support", style="yellow")
                    
                    for rule in rules_data['rules']:
                        table.add_row(
                            rule['rule'],
                            f"{rule['confidence']:.1%}",
                            f"{rule['lift']:.2f}",
                            f"{rule['support']:.3f}"
                        )
                    
                    console.print(table)
            elif choice == '6':
                console.print("\n🧪 Running Quick Test...")
                
                test_carts = [
                    ['bread', 'butter'],
                    ['milk', 'cereal'],
                    ['pasta', 'tomato_sauce']
                ]
                
                for cart in test_carts:
                    console.print(f"\n🛒 Testing cart: {cart}")
                    recommendations = client.get_recommendations(cart)
                    
                    if recommendations and recommendations.get('recommendations'):
                        recs = recommendations['recommendations'][:2]
                        for rec in recs:
                            console.print(f"  💡 {rec['item']} ({rec['confidence']:.1%})")
                    else:
                        console.print("  ❌ No recommendations")
                
                console.print("✅ Quick test complete!")
                
            elif choice == '0':
                break
            else:
                console.print("❌ Invalid choice", style="red")
                
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="yellow")
            break
        except Exception as e:
            console.print(f"❌ Error: {e}", style="red")

if __name__ == "__main__":
    main()
