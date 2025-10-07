"""CSC111 Project: Book Recommendation Graph Module

This module contains the graph implementation for representing book relationships
and connections, including the core graph data structure and book-specific
graph operations.

Copyright and Usage Information
===============================
This file is Copyright (c) 2025

Dorsa Rohani, Jiamei Huo, Ivy Huang, Behnaz Ghazanfari

"""
from __future__ import annotations
import pandas as pd
import ast
from typing import Dict, List, Optional, Set, Tuple, Any
import os

class Graph:
    """A class representing a graph with vertices and edges.
    
    Representation Invariants:
        - all(vertex in self._edges for vertex in self._vertices)
        - all(vertex in self._vertices for vertex in self._edges)
        - all(v2 in self._edges[v1] for v1 in self._edges for v2 in self._edges[v1] if v1 in self._edges[v2])
        - all(v1 not in self._edges[v1] for v1 in self._vertices)
    """
    _vertices: dict[str, Any]
    _edges: dict[str, set[str]]

    def __init__(self) -> None:
        """Initialize an empty graph."""
        self._vertices = {}
        self._edges = {}

    def add_vertex(self, v: str) -> None:
        """Add a vertex to this graph.
        
        Preconditions:
            - v not in self._vertices
        """
        if v not in self._vertices:
            self._vertices[v] = None
            self._edges[v] = set()

    def add_edge(self, v1: str, v2: str) -> None:
        """Add an edge between v1 and v2 to this graph.
        
        Preconditions:
            - v1 != v2
        """
        self.add_vertex(v1)
        self.add_vertex(v2)
        self._edges[v1].add(v2)
        self._edges[v2].add(v1)

    def connected(self, v1: str, v2: str) -> bool:
        """Return whether v1 and v2 are connected in this graph.
        
        Preconditions:
            - v1 in self._vertices and v2 in self._vertices
        """
        return v2 in self._edges.get(v1, set())

    def connected_path(self, v1: str, v2: str) -> Optional[list[str]]:
        """Return a path from v1 to v2 in this graph, or None if no path exists.
        
        Preconditions:
            - v1 and v2 are vertices in the graph
            
        >>> g = Graph()
        >>> g.add_edge('A', 'B')
        >>> g.add_edge('B', 'C')
        >>> g.connected_path('A', 'C')
        ['A', 'B', 'C']
        """
        if v1 not in self._vertices or v2 not in self._vertices:
            return None
            
        visited = {v1}
        path = [v1]
        
        def _dfs(current: str) -> bool:
            if current == v2:
                return True
                
            for neighbor in self._edges[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    if _dfs(neighbor):
                        return True
                    path.pop()
                    
            return False
            
        if _dfs(v1):
            return path
        return None

class BookGraph:
    """A graph based book recommendation system
    
    This class builds a graph where vertices are books and edges represent
    similarity between books based on shared genres and ratings
    
    Representation Invariants:
        - all(title in self.graph._vertices for title in self.books)
        - all(all(key in book for key in ['name', 'author', 'star_rating', 'num_ratings', 'num_reviews', 'genres', 'first_published']) for book in self.books.values())
        - all(book['star_rating'] >= 0 and book['num_ratings'] >= 0 and book['num_reviews'] >= 0 for book in self.books.values())
    """
    books: Dict[str, Dict]
    graph: Graph
    book_vertex_map: Dict[str, object]
    
    def __init__(self, csv_file: str) -> None:
        """Initialize a BookGraph from a .csv file of book data
        
        Preconditions:
            - os.path.exists(csv_file)
            - all(col in pd.read_csv(csv_file).columns for col in ['name', 'author', 'star_rating', 'num_ratings', 'num_reviews', 'genres', 'first_published'])
        """
        self.graph = Graph()
        self.books = {}
        self.book_vertex_map = {}
        self._load_books(csv_file)
        self._build_graph()
    
    def _load_books(self, csv_file: str) -> None:
        """Load books from .csv file.
        
        Preconditions:
            - csv_file exists and contains valid book data
            - csv_file has required columns: name, author, star_rating, etc.
        """
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            book_name = row['name']
            author = ast.literal_eval(row['author']) if isinstance(row['author'], str) else row['author']
            author_str = author[0] if isinstance(author, list) and len(author) > 0 else "Unknown"
            genres = ast.literal_eval(row['genres']) if isinstance(row['genres'], str) else row['genres']
            
            self.books[book_name] = {
                'name': book_name,
                'author': author_str,
                'star_rating': float(row['star_rating']) if not pd.isna(row['star_rating']) else 0.0,
                'num_ratings': int(float(row['num_ratings'])) if not pd.isna(row['num_ratings']) else 0,
                'num_reviews': int(float(row['num_reviews'])) if not pd.isna(row['num_reviews']) else 0,
                'genres': genres if isinstance(genres, list) else [],
                'first_published': row['first_published'] if not pd.isna(row['first_published']) else "Unknown"
            }
    
    def _build_graph(self) -> None:
        """Build the book recommendation graph.
        
        Creates vertices for each book and edges between books with shared genres.
        Edge weights are calculated based on genre similarity and review scores.
        """
        # add vertices for all books
        for title, book_data in self.books.items():
            self.graph.add_vertex(title)
            self.book_vertex_map[title] = title
        
        # connect books with shared genres
        book_titles = list(self.books.keys())
        for i in range(len(book_titles)):
            for j in range(i + 1, len(book_titles)):
                book1_title = book_titles[i]
                book2_title = book_titles[j]
                
                book1 = self.books[book1_title]
                book2 = self.books[book2_title]
                
                # calculate genre similarity
                genres1 = set(book1['genres'])
                genres2 = set(book2['genres'])
                
                shared_genres = genres1.intersection(genres2)
                
                # if the books share at least one genre, add an edge
                if shared_genres:
                    # calculate genre similarity coefficient
                    all_genres = genres1.union(genres2)
                    genre_similarity = len(shared_genres) / len(all_genres)
                    
                    # calculate review weighted scores 
                    score1 = book1['star_rating'] * book1['num_reviews']
                    score2 = book2['star_rating'] * book2['num_reviews']
                    
                    # calculate edge weight
                    edge_weight = (0.5 * (score1 + score2) * genre_similarity)
                    
                    # add edge to graph
                    self.graph.add_edge(book1_title, book2_title)
    
    def recommend_books(self, favorite_book: str, num_recommendations: int = 5) -> List[Dict]:
        """Return books similar to the given favorite book.
        
        Preconditions:
            - favorite_book in self.books
            - num_recommendations > 0
            
        Returns:
            A list of dictionaries containing book information for the top recommendations.
            
        >>> bg = BookGraph('test_data.csv')
        >>> recs = bg.recommend_books('Harry Potter', 3)
        >>> len(recs) <= 3
        True
        """
        if favorite_book not in self.books:
            return []
        
        # find all connected books
        connected_books = []
        for book_title in self.books:
            if book_title != favorite_book and self.graph.connected(favorite_book, book_title):
                path = self.graph.connected_path(favorite_book, book_title)
                
                # if theres a direct connection, calculate similarity
                if path and len(path) == 2:
                    fav_book = self.books[favorite_book]
                    potential_book = self.books[book_title]
                    
                    # calculate genre similarity
                    genres1 = set(fav_book['genres'])
                    genres2 = set(potential_book['genres'])
                    
                    shared_genres = genres1.intersection(genres2)
                    all_genres = genres1.union(genres2)
                    genre_similarity = len(shared_genres) / len(all_genres)
                    
                    # calculate weighted scores
                    score1 = fav_book['star_rating'] * fav_book['num_reviews']
                    score2 = potential_book['star_rating'] * potential_book['num_reviews']
                    
                    # calculate similarity score
                    similarity_score = (0.5 * (score1 + score2) * genre_similarity)
                    
                    connected_books.append((book_title, similarity_score))
        
        # sort by similarity score (descending)
        connected_books.sort(key=lambda x: x[1], reverse=True)
        
        # take top N recommendations
        top_recommendations = connected_books[:num_recommendations]
        
        # return book info for recommendations
        return [self.books[title] for title, _ in top_recommendations]
    
    def recommend_by_genres(self, preferred_genres: List[str], num_recommendations: int = 5) -> List[Dict]:
        """Return books based on a list of preferred genres.
        
        Preconditions:
            - len(preferred_genres) > 0
            - num_recommendations > 0
            
        Returns:
            A list of dictionaries containing book information for the top recommendations.
            
        >>> bg = BookGraph('test_data.csv')
        >>> recs = bg.recommend_by_genres(['Fantasy', 'Adventure'], 3)
        >>> len(recs) <= 3
        True
        """
        # convert input genres to lowercase for case-insensitive matching
        preferred_genres_lower = [genre.lower() for genre in preferred_genres]
        
        # score each book based on genre match and ratings
        scored_books = []
        for title, book_data in self.books.items():
            book_genres_lower = [genre.lower() for genre in book_data['genres']]
            
            # count matching genres
            matching_genres = set(preferred_genres_lower).intersection(book_genres_lower)
            if not matching_genres:
                continue
                
            # calculate match score based on genre overlap and book rating
            genre_match_score = len(matching_genres) / len(preferred_genres_lower)
            total_score = genre_match_score * book_data['star_rating']
            
            scored_books.append((title, total_score))
        
        # sort by score (descending)
        scored_books.sort(key=lambda x: x[1], reverse=True)
        
        # take top N recommendations
        top_recommendations = scored_books[:num_recommendations]
        
        # return book info for recommendations
        return [self.books[title] for title, _ in top_recommendations]

# if __name__ == '__main__':
    # import doctest
    # import python_ta
    
    # run doctests
    # doctest.testmod()
    
    # run PythonTA checks
    # python_ta.check_all(config={
    #     'extra-imports': ['pandas', 'ast', 'typing'],
    #     'allowed-io': [],  # no print/open/input calls in this module
    #     'max-line-length': 120,
    #     'disable': ['R0903', 'C0111']  # disable some warnings that are not critical
    # })
