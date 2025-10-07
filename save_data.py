"""CSC111 Project: Book Data Collection Module

This module handles downloading and saving book data from the Goodreads dataset
to a CSV file for use in the recommendation system.

Copyright and Usage Information
===============================
This file is Copyright (c) 2025

Dorsa Rohani, Jiamei Huo, Ivy Huang, Behnaz Ghazanfari

"""
import pandas as pd
from datasets import load_dataset


def save_data_to_csv(max_books: int, output_file: str) -> pd.DataFrame:
    """Download data from Hugging Face and save it to a .csv file.

    Preconditions:
        - max_books > 0
        - output_file.endswith('.csv')

    Returns:
        A pandas DataFrame containing the processed book data.

    >>> df = save_data_to_csv(10, 'test.csv')
    >>> len(df) <= 10
    True
    """
    # preconditions
    assert max_books > 0, "max_books must be positive"
    assert output_file.endswith('.csv'), "output_file must have .csv extension"

    # load the dataset from Hugging Face
    dataset = load_dataset("BrightData/Goodreads-Books", split="train")
    df = dataset.to_pandas()

    # take a sample of books with enough data
    # sort by number of reviews
    df['num_reviews'] = pd.to_numeric(df['num_reviews'], errors='coerce')
    df = df.sort_values(by='num_reviews', ascending=False)

    # necessary columns
    columns_to_keep = ['name', 'author', 'star_rating', 'num_ratings',
                       'num_reviews', 'genres', 'first_published']

    # get subset of columns that exist in the dataframe
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].head(max_books)

    # remove specific books by name
    books_to_remove = ['Educated', 'The Lightning Thief']
    df = df[~df['name'].isin(books_to_remove)]

    # save to .csv file
    df.to_csv(output_file, index=False)
    print(f"saved {len(df)} books to {output_file}")

    return df


if __name__ == '__main__':
    # import doctest
    # import python_ta

    # run doctests
    # doctest.testmod()

    # run PythonTA checks
    # python_ta.check_all(config={
    #     'extra-imports': ['pandas', 'datasets'],
    #     'allowed-io': ['save_data_to_csv'],  # function that calls print
    #     'max-line-length': 120,
    #     'disable': ['C0111']  # disable some warnings that are not critical
    # })

    # example usage
    save_data_to_csv(max_books=300, output_file='data.csv')
