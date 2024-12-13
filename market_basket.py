import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import plotly.graph_objects as go
from collections import defaultdict
import community
import random
from gensim.models import Word2Vec

class MarketBasketAnalyzer:
    def __init__(self, min_support=0.01, min_confidence=0.3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = None
        self.rules = None
        self.transaction_matrix = None
        self.product_encoder = LabelEncoder()
        self.graph = None
        self.product_embeddings = None

    def preprocess_transactions(self, transactions_df, transaction_id_col, product_col, 
                                timestamp_col=None, category_col=None):
        """
        Preprocess transaction data into a format suitable for market basket analysis.
        """
        # Create transaction lists
        transactions = transactions_df.groupby(transaction_id_col)[product_col].agg(list).values.tolist()

        # Generate product embeddings using Word2Vec
        self.product_embeddings = self._generate_product_embeddings(transactions)

        # Transform transactions to binary matrix
        te = TransactionEncoder()
        self.transaction_matrix = te.fit_transform(transactions)
        self.transaction_matrix = pd.DataFrame(self.transaction_matrix, columns=te.columns_)

        # Store additional metadata if available
        if timestamp_col and category_col:
            self.product_categories = transactions_df.groupby(product_col)[category_col].first().to_dict()
            self.temporal_patterns = self._analyze_temporal_patterns(transactions_df, 
                                                                     transaction_id_col, 
                                                                     product_col, 
                                                                     timestamp_col)

    def _generate_product_embeddings(self, transactions):
        """Generate product embeddings using Word2Vec."""
        model = Word2Vec(sentences=transactions, vector_size=50, window=5, min_count=1, workers=4)
        embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
        return embeddings

    def find_association_rules(self):
        """Generate association rules using Apriori algorithm."""
        self.frequent_itemsets = apriori(self.transaction_matrix, 
                                         min_support=self.min_support, 
                                         use_colnames=True)

        self.rules = association_rules(self.frequent_itemsets, 
                                       metric="confidence", 
                                       min_threshold=self.min_confidence)

        # Calculate additional metrics
        self.rules['leverage'] = self._calculate_leverage(self.rules)
        self.rules['conviction'] = self._calculate_conviction(self.rules)
        self.rules = self.rules.sort_values('lift', ascending=False)

    def _calculate_leverage(self, rules):
        """Calculate leverage metric for association rules"""
        return rules.apply(lambda x: (x['support'] - (x['antecedent support'] * x['consequent support'])), axis=1)

    def _calculate_conviction(self, rules):
        """Calculate conviction metric for association rules"""
        return rules.apply(lambda x: (1 - x['consequent support']) / (1 - x['confidence']) 
                           if x['confidence'] < 1 else np.inf, axis=1)

    def build_product_graph(self):
        """Create a weighted graph of product associations."""
        self.graph = nx.Graph()
        for _, rule in self.rules.iterrows():
            antecedents = list(rule['antecedents'])[0]
            consequents = list(rule['consequents'])[0]
            weight = rule['lift']
            self.graph.add_edge(antecedents, consequents, weight=weight)

    def identify_product_communities(self):
        """Detect communities of frequently co-purchased products."""
        if not self.graph:
            self.build_product_graph()
        communities = community.best_partition(self.graph)
        community_groups = defaultdict(list)
        for product, community_id in communities.items():
            community_groups[community_id].append(product)
        return community_groups

    def get_recommendations(self, basket, n_recommendations=5):
        """Get product recommendations based on current basket."""
        if not self.rules:
            raise ValueError("Association rules haven't been generated yet.")
        recommendations = defaultdict(float)

        for item in basket:
            relevant_rules = self.rules[self.rules['antecedents'].apply(lambda x: item in x)]
            for _, rule in relevant_rules.iterrows():
                consequent = list(rule['consequents'])[0]
                if consequent not in basket:
                    score = rule['lift'] * rule['confidence']
                    recommendations[consequent] = max(recommendations[consequent], score)

        return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

    def visualize_product_network(self, min_edge_weight=1.5):
        """Create interactive visualization of product network."""
        if not self.graph:
            self.build_product_graph()

        filtered_edges = [(u, v) for (u, v, d) in self.graph.edges(data=True) 
                          if d['weight'] >= min_edge_weight]
        subgraph = self.graph.edge_subgraph(filtered_edges)
        pos = nx.spring_layout(subgraph)

        edge_x, edge_y = [], []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=0.5, color='#888'),
                                hoverinfo='none',
                                mode='lines')

        node_x, node_y = [], []
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(x=node_x, y=node_y,
                                 mode='markers+text',
                                 hoverinfo='text',
                                 text=[str(node) for node in subgraph.nodes()],
                                 marker=dict(size=10, line_width=2))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Product Association Network',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig

    def _analyze_temporal_patterns(self, transactions_df, transaction_id_col, 
                                   product_col, timestamp_col):
        """Analyze temporal purchasing patterns."""
        if not pd.api.types.is_datetime64_any_dtype(transactions_df[timestamp_col]):
            transactions_df[timestamp_col] = pd.to_datetime(transactions_df[timestamp_col])

        transactions_df['hour'] = transactions_df[timestamp_col].dt.hour
        transactions_df['day_of_week'] = transactions_df[timestamp_col].dt.dayofweek

        return {
            'hourly_patterns': transactions_df.groupby('hour')[product_col].count(),
            'daily_patterns': transactions_df.groupby('day_of_week')[product_col].count()
        }

def generate_abstract_data(n_transactions=1000):
    """Generate abstract transaction data with innovative relationships."""
    # Define abstract product categories
    product_categories = {
        'Alpha': [f'P{i}' for i in range(1, 11)],   # High-frequency basics
        'Beta': [f'P{i}' for i in range(11, 21)],   # Mid-frequency complementary
        'Gamma': [f'P{i}' for i in range(21, 31)],  # Low-frequency luxury
        'Delta': [f'P{i}' for i in range(31, 41)],  # Seasonal/periodic
        'Epsilon': [f'P{i}' for i in range(41, 51)]  # Impulse/spontaneous
    }

    # Define relationship patterns
    relationship_patterns = [
        # High synergy pairs (70% co-occurrence)
        (['P1', 'P2'], 0.7),
        (['P11', 'P12'], 0.6),

        # Complementary triads (40% co-occurrence)
        (['P21', 'P22', 'P23'], 0.4),
        (['P31', 'P32', 'P33'], 0.4),

        # Chain patterns (if A->B then 30% chance of C)
        (['P41', 'P42'], 0.5, ['P43'], 0.3),

        # Exclusion patterns (rarely together)
        (['P5', 'P15'], 0.1)
    ]

    transactions = []
    start_date = datetime(2024, 1, 1)

    category_probabilities = {
        'Alpha': 0.6,    # High frequency
        'Beta': 0.4,     # Medium frequency
        'Gamma': 0.2,    # Low frequency
        'Delta': 0.3,    # Seasonal
        'Epsilon': 0.35  # Impulse
    }

    price_ranges = {
        'Alpha': (10, 30),
        'Beta': (30, 80),
        'Gamma': (80, 200),
        'Delta': (40, 120),
        'Epsilon': (20, 60)
    }

    for trans_id in range(n_transactions):
        items = set()

        # Apply relationship patterns
        for pattern in relationship_patterns:
            if len(pattern) == 2:
                products, probability = pattern
                if random.random() < probability:
                    items.update(products)
            elif len(pattern) == 4:
                initial_products, initial_prob, chain_products, chain_prob = pattern
                if random.random() < initial_prob:
                    items.update(initial_products)
                    if random.random() < chain_prob:
                        items.update(chain_products)

        # Add category-based products
        for category, products in product_categories.items():
            if random.random() < category_probabilities[category]:
                n_items = random.randint(1, 3)
                available = [p for p in products if p not in items]
                if available:
                    items.update(random.sample(available, min(n_items, len(available))))

        # Generate timestamp
        timestamp = start_date + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(8, 20),
            minutes=random.randint(0, 59)
        )

        # Create transaction records
        for item in items:
            category = next(cat for cat, products in product_categories.items() 
                            if item in products)
            transactions.append({
                'transaction_id': trans_id + 1,
                'product': item,
                'category': category,
                'timestamp': timestamp,
                'quantity': random.randint(1, 3),
                'unit_price': round(random.uniform(*price_ranges[category]), 2),
                'purchase_pattern': 'regular' if random.random() < 0.7 else 'promotional'
            })

    return pd.DataFrame(transactions)

def main():
    """Run complete market basket analysis example."""
    # Generate and save data
    print("Generating abstract transaction data...")
    data = generate_abstract_data(n_transactions=1000)
    data.to_csv('abstract_transactions.csv', index=False)
    print("Data saved to abstract_transactions.csv")

    # Initialize and run analysis
    analyzer = MarketBasketAnalyzer(min_support=0.01, min_confidence=0.1)

    print("\nPreprocessing transactions...")
    analyzer.preprocess_transactions(
        data,
        transaction_id_col='transaction_id',
        product_col='product',
        timestamp_col='timestamp',
        category_col='category'
    )

    print("\nFinding association rules...")
    analyzer.find_association_rules()

    # Analyze rules by category
    categories = data['category'].unique()
    for category in categories:
        category_products = data[data['category'] == category]['product'].unique()
        category_rules = analyzer.rules[
            analyzer.rules['antecedents'].apply(lambda x: any(p in x for p in category_products))
        ]
        if not category_rules.empty:
            print(f"\nTop rules for {category} products:")
            print(category_rules.head(3)[['antecedents', 'consequents', 'lift']])

    # Product type analysis
    print("\nProduct Type Analysis:")
    for category in categories:
        category_data = data[data['category'] == category]
        avg_quantity = category_data['quantity'].mean()
        avg_price = category_data['unit_price'].mean()
        print(f"\n{category} Products:")
        print(f"- Average quantity per transaction: {avg_quantity:.2f}")
        print(f"- Average unit price: ${avg_price:.2f}")

    # Generate recommendations
    print("\nSample Recommendations:")
    sample_baskets = [
        ['P1', 'P2'],     # High synergy pair
        ['P21', 'P22'],   # Part of triad
        ['P41', 'P42']    # Chain pattern
    ]

    for basket in sample_baskets:
        print(f"\nRecommendations for basket containing: {', '.join(basket)}")
        recommendations = analyzer.get_recommendations(basket)
        for product, score in recommendations:
            print(f"- {product}: {score:.3f}")

    # Create visualization
    print("\nCreating product network visualization...")
    fig = analyzer.visualize_product_network(min_edge_weight=1.2)
    fig.write_html("product_network.html")
    print("Visualization saved as 'product_network.html'")

    return analyzer, data

if __name__ == "__main__":
    analyzer, data = main()
