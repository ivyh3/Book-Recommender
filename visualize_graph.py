"""CSC111 Project: Book Data Visualization Module

This module handles the visualization of book data and recommendation system results,
providing interactive and informative visual representations of the data.

Copyright and Usage Information
===============================
This file is Copyright (c) 2025

Dorsa Rohani, Jiamei Huo, Ivy Huang, Behnaz Ghazanfari

"""

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from graph import BookGraph
from typing import Dict, List, Any
import os

class BookGraphVisualizer:
    """This class visualizes the book recommendation graph.
    Create an interactive network and visualization of book recommendations

    Instance Attributes:
        - book_graph: The BookGraph instance containing book data and connections
        - nx_graph: A NetworkX graph representation for visualization
        - user_preferences: Dictionary storing user genre preferences and other settings
        - recommendation_metrics: Dictionary storing evaluation metrics for recommendations

    Representation Invariants:
        - len(self.book_graph.books) > 0
        - all(isinstance(book_data['star_rating'], (int, float)) for book_data in self.book_graph.books.values())
        - all(isinstance(book_data['num_reviews'], int) for book_data in self.book_graph.books.values())
        - all(isinstance(book_data['genres'], list) for book_data in self.book_graph.books.values())
        - all(isinstance(book_data['author'], str) for book_data in self.book_graph.books.values())
        - all(0 <= book_data['star_rating'] <= 5 for book_data in self.book_graph.books.values())
        - all(book_data['num_reviews'] >= 0 for book_data in self.book_graph.books.values())
        - all(len(book_data['genres']) > 0 for book_data in self.book_graph.books.values())
        - all(isinstance(genre, str) for book_data in self.book_graph.books.values() for genre in book_data['genres'])
        - all(isinstance(edge[2]['weight'], (int, float)) for edge in self.nx_graph.edges(data=True))
        - all(edge[2]['weight'] >= 0 for edge in self.nx_graph.edges(data=True))
        - all(isinstance(pref, (list, float, str)) for pref in self.user_preferences.values())
        - all(isinstance(metric, float) for metric in self.recommendation_metrics.values())
        - all(0 <= metric <= 1 for metric in self.recommendation_metrics.values())
    """
    def __init__(self, data_file: str) -> None:
        """Initialize a BookGraphVisualizer from a .csv file of book data

        Preconditions:
            - os.path.exists(data_file)
            - data_file.endswith('.csv')
            - os.path.getsize(data_file) > 0

        Arguments:
            data_file: Path to the .csv file containing book data
        """
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} does not exist")
        if not data_file.endswith('.csv'):
            raise ValueError("Data file must be a CSV file")
        if os.path.getsize(data_file) == 0:
            raise ValueError("Data file cannot be empty")

        self.book_graph = BookGraph(data_file)
        self.nx_graph = self._create_networkx_graph()
        self.user_preferences = {'genres': [], 'min_rating': 0.0, 'recency': 'all'}
        self.recommendation_metrics = {'accuracy': 0.0, 'relevance': 0.0, 'diversity': 0.0}

        # verify representation invariants
        self._verify_representation_invariants()

    def _verify_representation_invariants(self) -> None:
        """Verify all representation invariants of the class.
        
        Raises:
            AssertionError: If any representation invariant is violated
        """
        assert len(self.book_graph.books) > 0, "Book graph must contain at least one book"
        assert all(isinstance(book_data['star_rating'], (int, float)) for book_data in self.book_graph.books.values()), \
            "All star ratings must be numeric"
        assert all(isinstance(book_data['num_reviews'], int) for book_data in self.book_graph.books.values()), \
            "All review counts must be integers"
        assert all(isinstance(book_data['genres'], list) for book_data in self.book_graph.books.values()), \
            "All genres must be stored as lists"
        assert all(isinstance(book_data['author'], str) for book_data in self.book_graph.books.values()), \
            "All authors must be strings"
        assert all(0 <= book_data['star_rating'] <= 5 for book_data in self.book_graph.books.values()), \
            "All star ratings must be between 0 and 5"
        assert all(book_data['num_reviews'] >= 0 for book_data in self.book_graph.books.values()), \
            "All review counts must be non-negative"
        assert all(len(book_data['genres']) > 0 for book_data in self.book_graph.books.values()), \
            "All books must have at least one genre"
        assert all(isinstance(genre, str) for book_data in self.book_graph.books.values() for genre in book_data['genres']), \
            "All genres must be strings"
        assert all(isinstance(edge[2]['weight'], (int, float)) for edge in self.nx_graph.edges(data=True)), \
            "All edge weights must be numeric"
        assert all(edge[2]['weight'] >= 0 for edge in self.nx_graph.edges(data=True)), \
            "All edge weights must be non-negative"
        assert all(isinstance(pref, (list, float, str)) for pref in self.user_preferences.values()), \
            "All user preferences must be of valid types"
        assert all(isinstance(metric, float) for metric in self.recommendation_metrics.values()), \
            "All metrics must be floats"
        assert all(0 <= metric <= 1 for metric in self.recommendation_metrics.values()), \
            "All metrics must be between 0 and 1"

    # ================================================
        # CREDIT: 
        # https://plotly.com/python/reference/index/
        # from lines 117 to 215 

    # ================================================
    def _create_networkx_graph(self) -> nx.Graph:
        """
        Return a NetworkX graph containing book nodes and weighted edges
        """
        G = nx.Graph()
        for title, book_data in self.book_graph.books.items():
            main_genres = book_data['genres'][:3] if len(book_data['genres']) >= 3 else book_data['genres']
            G.add_node(title, title=title, author=book_data['author'], rating=book_data['star_rating'],
                       genres=main_genres, reviews=book_data['num_reviews'])
        book_titles = list(self.book_graph.books.keys())
        for i in range(len(book_titles)):
            for j in range(i + 1, len(book_titles)):
                book1_title = book_titles[i]
                book2_title = book_titles[j]
                book1 = self.book_graph.books[book1_title]
                book2 = self.book_graph.books[book2_title]
                genres1 = set(book1['genres'])
                genres2 = set(book2['genres'])
                shared_genres = genres1.intersection(genres2)
                if len(shared_genres) > 0:
                    all_genres = genres1.union(genres2)
                    genre_similarity = len(shared_genres) / len(all_genres)
                    score1 = book1['star_rating'] * book1['num_reviews']
                    score2 = book2['star_rating'] * book2['num_reviews']
                    edge_weight = (0.5 * (score1 + score2) * genre_similarity)
                    G.add_edge(book1_title, book2_title, weight=edge_weight)
        return G


    def visualize_top_clusters(self, top_n: int = 2, max_neighbors: int = 30) -> None:
        """Visualize the top clusters of books in the network.

        Preconditions:
            - top_n > 0
            - max_neighbors > 0
            - top_n <= len(self.book_graph.books)
            - max_neighbors <= len(self.book_graph.books)

        Arguments:
            top_n: Number of top clusters to visualize
            max_neighbors: Maximum number of neighbors to show for each cluster
        """
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")
        if top_n > len(self.book_graph.books):
            raise ValueError("top_n cannot be larger than the number of books")
        if max_neighbors > len(self.book_graph.books):
            raise ValueError("max_neighbors cannot be larger than the number of books")

        G = self.nx_graph
        sorted_books = sorted(
            [(title, data['rating'] * data['reviews']) for title, data in G.nodes(data=True)],
            key=lambda x: x[1], reverse=True
        )
        top_books = [title for title, _ in sorted_books[:top_n]]
        fig = make_subplots(rows=1, cols=top_n, subplot_titles=top_books)

        for i, book in enumerate(top_books):
            neighbors = sorted(G[book].items(), key=lambda item: item[1]['weight'], reverse=True)
            selected_neighbors = [n for n, _ in neighbors[:max_neighbors]]

            #
            subgraph_nodes = [book] + selected_neighbors
            subG = G.subgraph(subgraph_nodes).copy()

            nodes_sorted = sorted(subG.nodes())
            edges_sorted = sorted(subG.edges(data=True))

            subG_sorted = nx.Graph()
            for node in nodes_sorted:
                subG_sorted.add_node(node, **subG.nodes[node])
            subG_sorted.add_edges_from(edges_sorted)

            subG = subG_sorted

            pos = nx.spring_layout(subG, seed=42, k=0.5)

            if book in pos:
                pos[book][0] += 5.0


            edge_x, edge_y = [], []
            for u, v in subG.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=0.3, color='rgba(70, 70, 70, 0.4)'),
                    hoverinfo='none'
                ),
                row=1, col=i + 1
            )

            node_x, node_y, hover_texts = [], [], []
            for n in subG.nodes():
                x, y = pos[n]
                node_x.append(x)
                node_y.append(y)
                d = subG.nodes[n]
                genres = ", ".join(d['genres'][:3]) if d['genres'] else "N/A"
                hover_texts.append(
                    f"<b>{d['title']}</b><br>Author: {d['author']}<br>Rating: {d['rating']:.2f}<br>Genres: {genres}")

            random.seed(42 + i)
            node_colors = [f'rgb({random.randint(40, 200)}, {random.randint(40, 200)}, {random.randint(40, 200)})' for _
                           in subG.nodes()]
            node_sizes = [min(20, 8 + (subG.nodes[n]['reviews'] / 1000)) for n in subG.nodes()]

            fig.add_trace(
                go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    marker=dict(
                        size=[s * 1.2 for s in node_sizes],
                        color=node_colors,
                        line=dict(width=2, color='white')
                    ),
                    text=hover_texts,
                    hoverinfo='text'
                ),
                row=1, col=i + 1
            )

        fig.update_layout(
            title="Top Book Clusters by Recommendation Score",
            height=600,
            showlegend=False,
            margin=dict(b=20, l=40, r=40, t=60),
            uirevision='static-layout'
        )

        for i in range(1, top_n + 1):
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=i)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=i)

        fig.show()

    def visualize_full_graph(self, max_books: int, min_edge_weight: float) -> None:
        """Return an interactive visualization of the book recommendation network

        Preconditions:
            - max_books > 0
            - min_edge_weight >= 0
            - max_books <= len(self.book_graph.books)

        Arguments:
            max_books: Max number of books to include in the visualization
            min_edge_weight: Min edge weight to include in the visualization
        """
        if max_books <= 0:
            raise ValueError("max_books must be positive")
        if min_edge_weight < 0:
            raise ValueError("min_edge_weight must be non-negative")
        if max_books > len(self.book_graph.books):
            raise ValueError("max_books cannot be larger than the number of books")

        # create a subgraph with the specified constraints
        subgraph = self._get_subgraph(max_books, min_edge_weight)

        # layout algorithm
        pos = nx.kamada_kawai_layout(subgraph)

        # create edge trace
        edge_x = []
        edge_y = []
        edge_weights = []

        for edge in subgraph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # store edge weight for colouring
            weight = edge[2]['weight']
            edge_weights.extend([weight, weight, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.0, color='rgba(50, 50, 50, 0.5)'),
            hoverinfo='none',
            mode='lines'
        )

        # create vertex trace
        node_x = []
        node_y = []

        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        # get node attributes for display
        node_titles = [subgraph.nodes[node]['title'] for node in subgraph.nodes()]
        node_authors = [subgraph.nodes[node]['author'] for node in subgraph.nodes()]
        node_ratings = [subgraph.nodes[node]['rating'] for node in subgraph.nodes()]
        node_sizes = [min(20 + (subgraph.nodes[node]['reviews'] / 1000), 50) for node in subgraph.nodes()]

        # make a mapping of main genres to colours
        all_genres = set()
        for node in subgraph.nodes():
            all_genres.update(subgraph.nodes[node]['genres'])
        # randomize colours
        genre_colors = {genre: f'rgb({random.randint(50, 200)}, {random.randint(50, 200)}, {random.randint(50, 200)})'
                       for genre in all_genres}

        # assign colours based on main genre
        node_colors = []
        for node in subgraph.nodes():
            if len(subgraph.nodes[node]['genres']) > 0:
                primary_genre = subgraph.nodes[node]['genres'][0]
                node_colors.append(genre_colors[primary_genre])
            else:
                node_colors.append('rgb(100, 100, 100)')  # default colour


        node_hover_text = []
        for i, node in enumerate(subgraph.nodes()):
            genres_text = ', '.join(subgraph.nodes[node]['genres'][:3])
            hover_text = f"Book: {node_titles[i]}<br>Author: {node_authors[i]}<br>Rating: {node_ratings[i]:.2f}<br>Genres: {genres_text}"
            node_hover_text.append(hover_text)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line_width=2,
                line=dict(color='white')
            ),
            text=node_hover_text
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                          title=dict(
                              text='Book Recommendation Network',
                              font=dict(size=16)
                          ),
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20, l=5, r=5, t=40),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          paper_bgcolor='rgba(240, 240, 240, 0.8)',
                          plot_bgcolor='rgba(240, 240, 240, 0.8)',
                       ))



        top_genres = sorted(all_genres, key=lambda g: sum(1 for node in subgraph.nodes() if g in subgraph.nodes[node]['genres']), reverse=True)[:10]

        for i, genre in enumerate(top_genres):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=genre_colors[genre]),
                name=genre,
                showlegend=True
            ))

        # displahy metrics if available
        if any(self.recommendation_metrics.values()):
            metrics_text = (
                f"Recommendation Metrics:<br>"
                f"Relevance: {self.recommendation_metrics['relevance']:.2f}<br>"
                f"Diversity: {self.recommendation_metrics['diversity']:.2f}<br>"
            )

            fig.add_annotation(
                x=0.05,
                y=0.05,
                xref="paper",
                yref="paper",
                text=metrics_text,
                showarrow=False,
                font=dict(size=12),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )

        fig.show()

     # ================================================
        # CREDIT: 
        # https://plotly.com/python/discrete-color/
        # https://plotly.com/python/subplots/#simple-subplot
        # from lines 419 to 592

    # ================================================

    def _get_subgraph(self, max_books: int, min_edge_weight: float) -> nx.Graph:
        """Return a subgraph of the book network for visualization

        Arguments:
            max_books: Maximum number of books to include
            min_edge_weight: Minimum edge weight to include
        """
        # filter edges by weight
        filtered_edges = [(u, v, d) for u, v, d in self.nx_graph.edges(data=True)
                         if d['weight'] >= min_edge_weight]

        # sort by edge weight
        filtered_edges.sort(key=lambda x: x[2]['weight'], reverse=True)

        # take top N edges
        top_edges = filtered_edges[:max_books*2]  # ensure we have enough edges

        # get unique vertices
        unique_nodes = set()
        for u, v, _ in top_edges:
            unique_nodes.add(u)
            unique_nodes.add(v)

        # limit to max_books
        if len(unique_nodes) > max_books:
            # get books with highest ratings/reviews
            book_scores = [(node, self.book_graph.books[node]['star_rating'] *
                           self.book_graph.books[node]['num_reviews'])
                          for node in unique_nodes]
            book_scores.sort(key=lambda x: x[1], reverse=True)
            top_nodes = {node for node, _ in book_scores[:max_books]}
        else:
            top_nodes = unique_nodes

        # create subgraph
        subgraph = self.nx_graph.subgraph(top_nodes).copy()

        return subgraph

    def visualize_book_connections(self, book_title: str, depth: int = 1) -> None:
        """Return connections from a specific book

        Preconditions:
            - book_title in self.book_graph.books
            - depth > 0
            - depth <= len(self.book_graph.books)

        Arguments:
            book_title: Title of the book to visualize connections for
            depth: Depth of connections to visualize (1 = direct connections only)
        """
        if book_title not in self.book_graph.books:
            raise ValueError(f"Book '{book_title}' not found in the dataset")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if depth > len(self.book_graph.books):
            raise ValueError("depth cannot be larger than the number of books")

        # create a subgraph with the book and its connections
        subgraph = nx.Graph()

        # add the central book
        book_data = self.book_graph.books[book_title]
        main_genres = book_data['genres'][:3] if len(book_data['genres']) >= 3 else book_data['genres']

        subgraph.add_node(book_title,
                         title=book_title,
                         author=book_data['author'],
                         rating=book_data['star_rating'],
                         genres=main_genres,
                         reviews=book_data['num_reviews'],
                         is_center=True)

        # add connected books up to the specified depth
        visited = {book_title}
        current_level = {book_title}

        
        for _ in range(depth):
            next_level = set()
            for current_book in current_level:
                for other_book in self.book_graph.books:
                    if other_book not in visited and self.book_graph.graph.connected(current_book, other_book):
                        path = self.book_graph.graph.connected_path(current_book, other_book)

                        if path and len(path) == 2:  # Direct connection
                            other_book_data = self.book_graph.books[other_book]
                            other_main_genres = other_book_data['genres'][:3] if len(other_book_data['genres']) >= 3 else other_book_data['genres']

                            subgraph.add_node(other_book,
                                             title=other_book,
                                             author=other_book_data['author'],
                                             rating=other_book_data['star_rating'],
                                             genres=other_main_genres,
                                             reviews=other_book_data['num_reviews'],
                                             is_center=False)

                            book1 = self.book_graph.books[current_book]
                            book2 = self.book_graph.books[other_book]

                            genres1 = set(book1['genres'])
                            genres2 = set(book2['genres'])

                            shared_genres = genres1.intersection(genres2)
                            all_genres = genres1.union(genres2)

                            genre_similarity = len(shared_genres) / len(all_genres)

                            if len(all_genres) > 0:
                                score1 = book1['star_rating'] * book1['num_reviews']
                                score2 = book2['star_rating'] * book2['num_reviews']

                                edge_weight = (0.5 * (score1 + score2) * genre_similarity)

                                subgraph.add_edge(current_book, other_book, weight=edge_weight)

                            visited.add(other_book)
                            next_level.add(other_book)

            current_level = next_level

        pos = nx.kamada_kawai_layout(subgraph)

        # create edge trace
        edge_x = []
        edge_y = []
        edge_weights = []

        for edge in subgraph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # store edge weight for coloring
            weight = edge[2]['weight']
            edge_weights.extend([weight, weight, None])

        # check if we have any edges before creating the trace
        if not edge_weights or all(w is None for w in edge_weights):
            # if no edges, create a simple line trace with default color
            edge_trace = go.Scatter(
                x=edge_x if edge_x else [None],
                y=edge_y if edge_y else [None],
                line=dict(width=1.0, color='rgba(66, 133, 244, 0.5)'),
                hoverinfo='none',
                mode='lines'
            )
        else:
            #  edge weights for coloring
            max_weight = max([w for w in edge_weights if w is not None])
            normalized_weights = [w/max_weight if w is not None else None for w in edge_weights]

            # create the line coloring based on normalized weights
            line_colors = []
            for w in normalized_weights:
                if w is None:
                    line_colors.append('rgba(66, 133, 244, 0.1)')
                else:
                    line_colors.append(f'rgba(66, 133, 244, {w})')

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1.5, color='rgba(66, 133, 244, 0.5)'),  # default colour
                hoverinfo='none',
                mode='lines'
            )

            # try to set custom colours if we have enough edges
            if len(line_colors) > 0:
                try:
                    edge_trace.line.color = line_colors
                except:
                    # if custom coloring fails, we'll use the default color set above
                    pass

        # create node traces - one for regular nodes, one for the central node
        center_x, center_y = [], []
        regular_x, regular_y = [], []

        center_trace_data = None
        regular_trace_data = []

        for node in subgraph.nodes():
            x, y = pos[node]

            is_center = node == book_title

            if is_center:
                center_x.append(x)
                center_y.append(y)

                # extract node attributes for the center node
                center_title = subgraph.nodes[node]['title']
                center_author = subgraph.nodes[node]['author']
                center_rating = subgraph.nodes[node]['rating']
                center_genres = ', '.join(subgraph.nodes[node]['genres'][:3])

                center_hover_text = f"Book: {center_title}<br>Author: {center_author}<br>Rating: {center_rating:.2f}<br>Genres: {center_genres}"

                center_trace_data = {
                    'x': center_x,
                    'y': center_y,
                    'text': center_hover_text,
                    'size': 30,
                    'color': 'rgb(255, 0, 0)'  # red for central node
                }
            else:
                regular_x.append(x)
                regular_y.append(y)

                # extract node attributes
                node_title = subgraph.nodes[node]['title']
                node_author = subgraph.nodes[node]['author']
                node_rating = subgraph.nodes[node]['rating']
                node_genres = ', '.join(subgraph.nodes[node]['genres'][:3])

                hover_text = f"Book: {node_title}<br>Author: {node_author}<br>Rating: {node_rating:.2f}<br>Genres: {node_genres}"

                # get color based on primary genre
                primary_genre = subgraph.nodes[node]['genres'][0] if subgraph.nodes[node]['genres'] else None
                color = f'rgb({random.randint(50, 200)}, {random.randint(50, 200)}, {random.randint(50, 200)})'

                regular_trace_data.append({
                    'x': [x],
                    'y': [y],
                    'text': hover_text,
                    'size': min(15 + (subgraph.nodes[node]['reviews'] / 1000), 25),
                    'color': color
                })

        # create traces
        traces = []

        # add edge trace
        traces.append(edge_trace)

        # add center node trace
        if center_trace_data:
            center_trace = go.Scatter(
                x=center_trace_data['x'],
                y=center_trace_data['y'],
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    size=center_trace_data['size'],
                    color=center_trace_data['color'],
                    line_width=2,
                    line=dict(color='white')
                ),
                text=center_trace_data['text'],
                name=book_title
            )
            traces.append(center_trace)

        # add regular node traces
        for i, node_data in enumerate(regular_trace_data):
            node_trace = go.Scatter(
                x=node_data['x'],
                y=node_data['y'],
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    size=node_data['size'],
                    color=node_data['color'],
                    line_width=2,
                    line=dict(color='white')
                ),
                text=node_data['text'],
                showlegend=False
            )
            traces.append(node_trace)

        # create figure
        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=dict(
                    text=f'Books Similar to "{book_title}"',
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='rgba(240, 240, 240, 0.8)',
                plot_bgcolor='rgba(240, 240, 240, 0.8)',
            )
        )

        fig.show()

    def visualize_genre_distribution(self) -> None:
        """Visualize the distribution of genres in the datas et"""
        # count genre occurrences
        genre_counts = {}
        for book_data in self.book_graph.books.values():
            for genre in book_data['genres']:
                if genre in genre_counts:
                    genre_counts[genre] += 1
                else:
                    genre_counts[genre] = 1

        # sort genres by count
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

        # take top 20 genres
        top_genres = sorted_genres[:20]

        # create bar chart
        genres = [g[0] for g in top_genres]
        counts = [g[1] for g in top_genres]

        fig = go.Figure(data=[
            go.Bar(x=genres, y=counts,
                  marker_color='rgb(66, 133, 244)')
        ])

        fig.update_layout(
            title='Top 20 Genres in the Dataset',
            xaxis_title='Genre',
            yaxis_title='Number of Books',
            xaxis_tickangle=-45
        )

        fig.show()

    def visualize_rating_distribution(self) -> None:
        """Set the distribution of book ratings in the dataset."""
        ratings = [book_data['star_rating'] for book_data in self.book_graph.books.values()]

        fig = go.Figure(data=[
            go.Histogram(x=ratings,
                        nbinsx=20,
                        marker_color='rgb(66, 133, 244)')
        ])

        fig.update_layout(
            title='Distribution of Book Ratings',
            xaxis_title='Rating',
            yaxis_title='Number of Books'
        )

        fig.show()

    def set_user_preferences(self, preferred_genres: list, min_rating: float = 0.0, recency: str = 'all') -> None:
        """Set user preferences for book recommendations

        Preconditions:
            - all(isinstance(genre, str) for genre in preferred_genres)
            - 0 <= min_rating <= 5
            - recency in {'all', 'recent', 'classic'}

        Arguments:
            preferred_genres: List of genres the user is interested in
            min_rating: Minimum star rating for recommended books
            recency: Filter for book recency ('all', 'recent', 'classic')
        """
        if not all(isinstance(genre, str) for genre in preferred_genres):
            raise TypeError("All genres must be strings")
        if not 0 <= min_rating <= 5:
            raise ValueError("min_rating must be between 0 and 5")
        if recency not in {'all', 'recent', 'classic'}:
            raise ValueError("recency must be one of: 'all', 'recent', 'classic'")

        self.user_preferences['genres'] = preferred_genres
        self.user_preferences['min_rating'] = min_rating
        self.user_preferences['recency'] = recency
        print(f"User preferences set: {self.user_preferences}")

    def evaluate_recommendations(self, recommended_books: list, user_feedback: list = None) -> dict:
        """Return the quality of book recommendations

        Preconditions:
            - all(book in self.book_graph.books for book in recommended_books)
            - user_feedback is None or len(user_feedback) == len(recommended_books)
            - user_feedback is None or all(0 <= rating <= 5 for rating in user_feedback)

        Arguments:
            recommended_books: List of recommended book titles
            user_feedback: Optional list of user ratings for recommendations

        Returns:
            Dictionary containing evaluation metrics
        """
        if not all(book in self.book_graph.books for book in recommended_books):
            raise ValueError("All recommended books must exist in the dataset")
        if user_feedback is not None:
            if len(user_feedback) != len(recommended_books):
                raise ValueError("user_feedback length must match recommended_books length")
            if not all(0 <= rating <= 5 for rating in user_feedback):
                raise ValueError("All user feedback ratings must be between 0 and 5")

        metrics = {
            'accuracy': 0.0,
            'relevance': 0.0,
            'diversity': 0.0
        }

        # calculate genre relevance (match between recommendations and preferences)
        if self.user_preferences['genres']:
            genre_matches = 0
            for book in recommended_books:
                if book in self.book_graph.books:
                    book_genres = set(self.book_graph.books[book]['genres'])
                    preferred_genres = set(self.user_preferences['genres'])
                    if book_genres.intersection(preferred_genres):
                        genre_matches += 1

            metrics['relevance'] = genre_matches / len(recommended_books) if recommended_books else 0.0

        # calculate diversity (unique genres among recommendations)
        all_genres = set()
        for book in recommended_books:
            if book in self.book_graph.books:
                all_genres.update(self.book_graph.books[book]['genres'])

        metrics['diversity'] = len(all_genres) / 10  # normalize by assuming 10 genres is maximum diversity

        # if user feedback is provided, calculate accuracy
        if user_feedback:
            # assuming user_feedback is a list of ratings for each recommendation
            if len(user_feedback) == len(recommended_books):
                metrics['accuracy'] = sum(user_feedback) / len(user_feedback) / 5.0  # normalize to 0-1

        self.recommendation_metrics = metrics
        return metrics

    def get_recommendations_for_user(self, num_recommendations: int = 10) -> list:
        """Return personalized book recommendations based on user preferences

        Preconditions:
            - num_recommendations > 0
            - num_recommendations <= len(self.book_graph.books)

        Arguments:
            num_recommendations: Number of books to recommend
        """
        if num_recommendations <= 0:
            raise ValueError("num_recommendations must be positive")
        if num_recommendations > len(self.book_graph.books):
            raise ValueError("num_recommendations cannot be larger than the number of books")

        # filter books by user preferences
        candidates = []

        for title, book_data in self.book_graph.books.items():
            # check if book meets minimum rating
            if book_data['star_rating'] < self.user_preferences['min_rating']:
                continue

            # check if book matches genre preferences
            if self.user_preferences['genres']:
                book_genres = set(book_data['genres'])
                preferred_genres = set(self.user_preferences['genres'])

                # if no genre overlap, skip this book
                if not book_genres.intersection(preferred_genres):
                    continue

            # add book to candidates with a score
            genre_match_score = 0
            if self.user_preferences['genres']:
                book_genres = set(book_data['genres'])
                preferred_genres = set(self.user_preferences['genres'])
                genre_match_score = len(book_genres.intersection(preferred_genres)) / len(preferred_genres)

            # combined score based on ratings and genre match
            score = (0.5 * book_data['star_rating'] / 5.0) + (0.5 * genre_match_score)

            candidates.append((title, score))

        # sort by score and take top recommendations
        candidates.sort(key=lambda x: x[1], reverse=True)
        recommendations = [title for title, _ in candidates[:num_recommendations]]

        # evaluate these recommendations
        self.evaluate_recommendations(recommendations)

        return recommendations

    #######
    def visualize_top_clusters_from_books(self, book_list: list, max_neighbors: int = 30) -> None:
        """Visualize clusters for a list of books.

        Preconditions:
            - len(book_list) > 0
            - all(book in self.book_graph.books for book in book_list)
            - max_neighbors > 0
            - max_neighbors <= len(self.book_graph.books)

        Arguments:
            book_list: List of book titles to visualize clusters for
            max_neighbors: Maximum number of neighbors to show for each cluster
        """
        if not book_list:
            raise ValueError("book_list cannot be empty")
        if not all(book in self.book_graph.books for book in book_list):
            raise ValueError("All books in book_list must exist in the dataset")
        if max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")
        if max_neighbors > len(self.book_graph.books):
            raise ValueError("max_neighbors cannot be larger than the number of books")

        G = self.nx_graph
        fig = make_subplots(rows=1, cols=min(2, len(book_list)), subplot_titles=book_list[:2])

        for i, book in enumerate(book_list[:2]):
            neighbors = sorted(G[book].items(), key=lambda item: item[1]['weight'], reverse=True)
            selected_neighbors = [n for n, _ in neighbors[:max_neighbors]]
            subgraph_nodes = [book] + selected_neighbors
            subG = G.subgraph(subgraph_nodes).copy()
            nodes_sorted = sorted(subG.nodes())
            edges_sorted = sorted(subG.edges(data=True))

            subG_sorted = nx.Graph()
            for node in nodes_sorted:
                subG_sorted.add_node(node, **subG.nodes[node])
            subG_sorted.add_edges_from(edges_sorted)

            subG = subG_sorted
            pos = nx.spring_layout(subG, seed=42, k=0.5)
            if book in pos:
                pos[book][1] += 3.0

            edge_x, edge_y = [], []
            for u, v in subG.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=0.3, color='rgba(70, 70, 70, 0.4)'),
                hoverinfo='none'),
                row=1, col=i + 1
            )

            node_x, node_y, hover_texts = [], [], []
            for n in subG.nodes():
                x, y = pos[n]
                node_x.append(x)
                node_y.append(y)
                d = subG.nodes[n]
                genres = ", ".join(d['genres'][:3]) if d['genres'] else "N/A"
                hover_texts.append(
                    f"<b>{d['title']}</b><br>Author: {d['author']}<br>Rating: {d['rating']:.2f}<br>Genres: {genres}"
                )

                random.seed(42 + i)
                node_colors = [
                    f'rgb({random.randint(40, 200)}, {random.randint(40, 200)}, {random.randint(40, 200)})'
                    for _ in subG.nodes()
                ]
                node_sizes = [
                    min(20, 8 + (subG.nodes[n]['reviews'] / 1000))
                    for n in subG.nodes()
                ]

                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    marker=dict(size=[s * 1.2 for s in node_sizes], color=node_colors,
                                line=dict(width=2, color='white')),
                    text=hover_texts,
                    hoverinfo='text'),
                    row=1, col=i + 1
                )

        fig.update_layout(
            title="Top Book Clusters by Recommendation Score (Filtered)",
        height = 600,
        showlegend = False,
        margin = dict(b=20, l=40, r=40, t=60),
        uirevision='static-layout'
        )

        for i in range(1, min(2, len(book_list)) + 1):
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=i, uirevision='static-layout')
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=i, uirevision='static-layout')


        fig.show()


def visualize_book_network(data_file: str = 'data.csv', max_books: int = 100,
                         user_genres: list = None, min_rating: float = 0.0) -> None:
    """Return the book recommendation network.

    Arguments:
        data_file: Path to the CSV file containing book data
        max_books: Maximum number of books to visualize
        user_genres: Optional list of genres to filter books by
        min_rating: Minimum rating for books to include
    """
    visualizer = BookGraphVisualizer(data_file)

    # set user preferences if provided
    if user_genres:
        visualizer.set_user_preferences(user_genres, min_rating)

    visualizer.visualize_full_graph(max_books=max_books, min_edge_weight=0)


def visualize_book_connections(book_title: str, data_file: str = 'data.csv') -> None:
    """Visualize connections for a specific book.

    Arguments:
        book_title: Title of the book to visualize connections for
        data_file: Path to the .csv file containing book data
    """
    visualizer = BookGraphVisualizer(data_file)
    visualizer.visualize_book_connections(book_title)


def visualize_genre_distribution(data_file: str = 'data.csv') -> None:
    """Return the distribution of genres in the dataset.

    Arguments:
        data_file: Path to the .csv file containing book data
    """
    visualizer = BookGraphVisualizer(data_file)
    visualizer.visualize_genre_distribution()


def visualize_rating_distribution(data_file: str = 'data.csv') -> None:
    """Visualize the distribution of book ratings in the dataset.

    Args:
        data_file: Path to the .csv file containing book data
    """
    visualizer = BookGraphVisualizer(data_file)
    visualizer.visualize_rating_distribution()

# if __name__ == '__main__':
    # import doctest
    # import python_ta
    
    # run doctests
    # doctest.testmod()
    
    # run PythonTA checks
    # python_ta.check_all(config={
    #     'extra-imports': ['networkx', 'plotly.graph_objects', 'plotly.subplots', 'random', 'graph', 'typing', 'os'],
    #     'allowed-io': ['set_user_preferences', 'evaluate_recommendations'],  # functions that call print
    #     'max-line-length': 120,
    #     'disable': ['C0111', 'R0903', 'R0913', 'R0914']  # disable some warnings that are not critical
    # })