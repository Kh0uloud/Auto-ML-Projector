from dateutil import parser
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import pandas as pd
def convert_to_hex(color) -> str:
    """Normalizes color values to 6-character lowercase hex strings."""
    if pd.isna(color) or color in [np.inf, -np.inf]:
        return 'ffffff'
    elif isinstance(color, (int, float)):
        try:
            return f'{int(color):06x}'
        except Exception as e:
            logging.warning(f'Error converting color {color}: {e}')
            return 'ffffff'
    elif isinstance(color, str):
        return color.lower().strip().lstrip('#')
    else:
        return 'ffffff'

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Converts a hex color code to an (R, G, B) tuple."""
    if isinstance(hex_color, str):
        hex_color = hex_color.strip().lstrip('#')
        if len(hex_color) == 6 and all((c in '0123456789abcdefABCDEF' for c in hex_color)):
            return tuple((int(hex_color[i:i + 2], 16) for i in (0, 2, 4)))
    return (0, 0, 0)

def normalize_hex_color_columns(df: pd.DataFrame, color_columns: list[str]) -> pd.DataFrame:
    """Applies convert_to_hex to each color column to ensure consistent formatting."""
    for col in color_columns:
        df[col] = df[col].apply(convert_to_hex)
    return df

def expand_hex_color_columns(df: pd.DataFrame, color_columns: list[str]) -> pd.DataFrame:
    """Expands hex color columns into RGB tuple columns (r, g, b)."""
    for col in color_columns:
        rgb_cols = [f'{col}_r', f'{col}_g', f'{col}_b']
        df[rgb_cols] = df[col].apply(lambda x: pd.Series(hex_to_rgb(x)))
    return df

def calculate_brightness(r: int, g: int, b: int) -> float:
    """Computes perceived brightness from RGB values using luminance formula."""
    return 0.299 * r + 0.587 * g + 0.114 * b

def add_color_brightness_columns(df: pd.DataFrame, color_columns: list[str]) -> pd.DataFrame:
    """Adds brightness column for each RGB color triplet in specified color columns."""
    for col in color_columns:
        df[f'{col}_brightness'] = df.apply(lambda row: calculate_brightness(row.get(f'{col}_r', 0), row.get(f'{col}_g', 0), row.get(f'{col}_b', 0)), axis=1)
    return df

def clean_source(df):
    df['source'] = df['source'].str.extract('>(.*?)<')
    source_counts = df.groupby('user_id')['source'].nunique().reset_index()
    source_counts.rename(columns={'source': 'unique_sources'}, inplace=True)
    return (df, source_counts)

def compute_aggregates(df):
    return df.groupby('user_id').agg(total_retweets=('retweet_count', 'sum'), total_favorites=('favorite_count', 'sum'), avg_retweets=('retweet_count', 'mean'), avg_favorites=('favorite_count', 'mean'), tweet_count=('text', 'count')).reset_index()

def compute_tweet_frequency(df):
    df_sorted = df.sort_values(by=['user_id', 'created_at'])
    df_sorted['time_diff'] = df_sorted.groupby('user_id')['created_at'].diff().dt.total_seconds()
    tweet_freq = df_sorted.groupby('user_id').agg(mean_time_gap=('time_diff', 'mean'), median_time_gap=('time_diff', 'median'), std_time_gap=('time_diff', 'std'), total_timespan=('created_at', lambda x: (x.max() - x.min()).total_seconds())).reset_index()
    tweet_counts = df_sorted.groupby('user_id')['text'].count()
    tweet_freq['tweet_rate'] = tweet_freq.progress_apply(lambda row: 1 if row['total_timespan'] == 0 else tweet_counts.get(row['user_id'], 0) / (row['total_timespan'] / 86400), axis=1)
    return tweet_freq

def compute_quote_features(df):
    quote_features = df.groupby('user_id').agg(total_quotes=('is_quote_status', 'sum'), quote_ratio=('is_quote_status', 'mean')).reset_index()
    quoted_interactions = df[df['quoted_status_id'].notna()].merge(df[['user_id', 'depression']].rename(columns={'user_id': 'quoted_status_id', 'depression': 'quoted_depressed'}), on='quoted_status_id', how='left')
    quoted_counts = quoted_interactions.groupby('user_id')['quoted_depressed'].sum().reset_index()
    quoted_counts.rename(columns={'quoted_depressed': 'quotes_from_depressed'}, inplace=True)
    total_quotes = quoted_interactions.groupby('user_id')['quoted_depressed'].count().reset_index()
    total_quotes.rename(columns={'quoted_depressed': 'total_quotes'}, inplace=True)
    quoted_counts = quoted_counts.merge(total_quotes, on='user_id', how='left')
    quoted_counts['quotes_from_non_depressed'] = quoted_counts['total_quotes'] - quoted_counts['quotes_from_depressed']
    return quote_features.merge(quoted_counts, on='user_id', how='left')

def compute_interaction_features(df):
    interactions = df.groupby('user_id')['in_reply_to_user_id'].nunique().reset_index()
    interactions.rename(columns={'in_reply_to_user_id': 'unique_interactions'}, inplace=True)
    depressed_interactions = df[df['in_reply_to_user_id'].notna()].merge(df[['user_id', 'depression']].rename(columns={'user_id': 'in_reply_to_user_id', 'depression': 'reply_depressed'}), on='in_reply_to_user_id', how='left')
    depressed_counts = depressed_interactions.groupby('user_id')['reply_depressed'].sum().reset_index()
    depressed_counts.rename(columns={'reply_depressed': 'interactions_with_depressed'}, inplace=True)
    total_interactions = depressed_interactions.groupby('user_id')['reply_depressed'].count().reset_index()
    total_interactions.rename(columns={'reply_depressed': 'total_interactions'}, inplace=True)
    depressed_counts = depressed_counts.merge(total_interactions, on='user_id', how='left')
    depressed_counts['interactions_with_non_depressed'] = depressed_counts['total_interactions'] - depressed_counts['interactions_with_depressed']
    return interactions.merge(depressed_counts, on='user_id', how='left')

