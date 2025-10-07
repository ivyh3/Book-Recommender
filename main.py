"""CSC111 Project: Book Recommendation System Main Module

This module contains the main execution flow for the book recommendation system,
including data processing, user interaction, and visualization.

Copyright and Usage Information
===============================
This file is Copyright (c) 2025

Dorsa Rohani, Jiamei Huo, Ivy Huang, Behnaz Ghazanfari

"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import random
import os

from visualize_graph import BookGraphVisualizer, visualize_book_network, visualize_book_connections, visualize_genre_distribution, visualize_rating_distribution
from save_data import save_data_to_csv


def process_and_validate_data(data_file: str) -> str:
    """Process and validate book data from a CSV file.
    
    Preconditions:
        - os.path.exists(data_file)
        - all(col in pd.read_csv(data_file).columns for col in ['name', 'author', 'star_rating', 'num_ratings', 'num_reviews', 'genres', 'first_published'])
        
    Returns:
        Path to the processed data file.
        
    >>> processed = process_and_validate_data('test_data.csv')
    >>> processed.endswith('_processed.csv')
    True
    """
    df = pd.read_csv(data_file)

    print(f"Loaded {len(df)} book records")

    # check for and handle missing values
    missing_titles = df['name'].isnull().sum()
    missing_authors = df['author'].isnull().sum()
    missing_ratings = df['star_rating'].isnull().sum()

    if missing_titles > 0 or missing_authors > 0:
        print(f"Warning: Found missing data - {missing_titles} titles, {missing_authors} authors")
        # drop rows with missing essential data
        df = df.dropna(subset=['name', 'author'])
        print(f"Filtered to {len(df)} valid records")

    if missing_ratings > 0:
        print(f"Filling {missing_ratings} missing ratings with average value")
        df['star_rating'].fillna(df['star_rating'].mean(), inplace=True)

    # check for duplicate titles and handle them
    duplicate_titles = df['name'].duplicated().sum()
    if duplicate_titles > 0:
        print(f"Found {duplicate_titles} duplicate book titles. Keeping first occurrence.")
        df = df.drop_duplicates(subset=['name'])

    # ensure numeric columns are properly typed
    df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
    df['num_reviews'] = pd.to_numeric(df['num_reviews'], errors='coerce')

    # save processed data to temporary file
    processed_file = data_file.replace('.csv', '_processed.csv')
    df.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")

    return processed_file


def get_user_input_preferences() -> tuple[list[str], float, str]:
    """Return user preferences through terminal input.
    
    Returns:
        A tuple containing:
        - list of preferred genres
        - minimum rating (0-5)
        - recency preference ('all', 'recent', or 'classic')
        
    >>> prefs = get_user_input_preferences()
    >>> isinstance(prefs[0], list)
    True
    >>> 0 <= prefs[1] <= 5
    True
    """
    print("\nBook Interactive and Visualized Recommendation System!")
    print("============================================")

    # get genre preferences
    print("\nSelect your preferred genres (comma-separated numbers):")
    available_genres = ['Fantasy', 'Science Fiction', 'Mystery', 'Romance',
                       'Thriller', 'Biography', 'History', 'Fiction',
                       'Non-Fiction', 'Young Adult', 'Horror', 'Adventure']

    for i, genre in enumerate(available_genres, 1):
        print(f"{i}. {genre}")

    genre_input = input("\nEnter your choices (e.g., 1,3,5): ")
    selected_genres = []

    try:
        genre_indices = [int(idx.strip()) - 1 for idx in genre_input.split(',')]
        selected_genres = [available_genres[idx] for idx in genre_indices if 0 <= idx < len(available_genres)]
    except:
        print("Invalid input. No genres selected.")

    print(f"Selected genres: {', '.join(selected_genres) if selected_genres else 'None'}")

    # get minimum rating preference
    min_rating = 0.0
    rating_input = input("\nEnter minimum book rating (0-5, press Enter for no minimum): ")

    try:
        if rating_input.strip():
            min_rating = float(rating_input.strip())
            min_rating = max(0, min(5, min_rating))  # ensure between 0-5
    except:
        print("Invalid rating. Using no minimum rating.")

    print(f"Minimum rating set to: {min_rating}")

    return (selected_genres, min_rating, 'all')  # 'all' for recency by default


def simple_recommendation_flow(data_file: str = 'data.csv') -> None:
    """Run a simple recommendation flow with user interaction.
    
    Preconditions:
        - os.path.exists(data_file)
        - os.path.exists(data_file.replace('.csv', '_processed.csv'))
    """
    print("\nWelcome to Book Recommendation System")
    print("===================================")

    # process data
    processed_data = process_and_validate_data(data_file)
    visualizer = BookGraphVisualizer(processed_data)

    # get user preferences
    selected_genres, min_rating, recency = get_user_input_preferences()

    # set preferences in visualizer
    visualizer.set_user_preferences(selected_genres, min_rating, recency)

    # use default number of recommendations
    num_recommendations = 10

    # get recommendations
    recommendations = visualizer.get_recommendations_for_user(num_recommendations)

    # display recommendations
    print("\nYour Personalized Book Recommendations:")
    print("---------------------------------------")
    if recommendations:
        for i, book in enumerate(recommendations, 1):
            book_data = visualizer.book_graph.books[book]
            print(f"{i}. {book} by {book_data['author']} - Rating: {book_data['star_rating']:.2f}")
            print(f"   Genres: {', '.join(book_data['genres'][:3])}")

        visualizer.visualize_top_clusters_from_books(recommendations, max_neighbors=10)

        # display metrics
        metrics = visualizer.recommendation_metrics
        print("\nRecommendation Quality Metrics:")
        print(f"Relevance (match to your preferences): {metrics['relevance']:.2f}")
        print(f"Diversity (variety of genres): {metrics['diversity']:.2f}")

    else:
        print("No recommendations found matching your preferences.")
        print("Try selecting different genres or lowering the minimum rating.")


if __name__ == '__main__':
    """
    Book Recommendation System - Main Execution
    
    Runs a simple recommendation flow with user interaction
    """
    # import doctest
    # import python_ta
    
    # # Run doctests
    # doctest.testmod()
    
    # # Run PythonTA checks
    # python_ta.check_all(config={
    #     'extra-imports': ['pandas', 'plotly.graph_objects', 'plotly.subplots', 'networkx', 'random', 'os'],
    #     'allowed-io': ['process_and_validate_data', 'get_user_input_preferences', 'simple_recommendation_flow'],  # functions that call print/input
    #     'max-line-length': 120,
    #     'disable': ['C0111', 'R0903', 'R0913', 'R0914']  # disable some warnings that are not critical
    # })

    print("Book Recommender")
    print("================================")
    
    # process data and get recommendations
    processed_data = process_and_validate_data('data.csv')
    visualizer = BookGraphVisualizer(processed_data)
    
    # get user preferences and recommendations
    selected_genres, min_rating, recency = get_user_input_preferences()
    visualizer.set_user_preferences(selected_genres, min_rating, recency)
    recommendations = visualizer.get_recommendations_for_user(num_recommendations=10)
    
    # display results
    print("\nYour Personalized Book Recommendations:")
    print("---------------------------------------")
    if recommendations:
        for i, book in enumerate(recommendations, 1):
            book_data = visualizer.book_graph.books[book]
            print(f"{i}. {book} by {book_data['author']} - Rating: {book_data['star_rating']:.2f}")
            print(f"   Genres: {', '.join(book_data['genres'][:3])}")
        
        # show recommendation clusters
        print("\nVisualizing recommendation clusters...")
        visualizer.visualize_top_clusters_from_books(recommendations, max_neighbors=10)
        
        # display metrics
        metrics = visualizer.recommendation_metrics
        print("\nRecommendation Quality Metrics:")
        print(f"Relevance (match to your preferences): {metrics['relevance']:.2f}")
        print(f"Diversity (variety of genres): {metrics['diversity']:.2f}")
    else:
        print("No recommendations found matching your preferences.")
        print("Try selecting different genres or lowering the minimum rating.")
    
    print("\nThank you for using the Book Recommendator!")

    