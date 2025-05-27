from src.train import prepare_data
from src.train import optimize_models
from src.features import compute_interaction_features
from src.features import compute_quote_features
from src.features import compute_tweet_frequency
from src.features import compute_aggregates
from src.features import clean_source
from src.visualize import plot_correlation_heatmap
from src.visualize import plot_feature_bars
from src.features import add_color_brightness_columns
from src.features import expand_hex_color_columns
from src.features import normalize_hex_color_columns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import logging
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import pandas as pd
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
from dateutil import parser
import numpy as np
import eli5
import seaborn as sns
import os
# Auto-generated pipeline entry point


# ingest_data
home = os.getcwd()
print(home)
users_data_path = os.path.join(home, 'data', 'sample_users.csv')
timeline_data_path = os.path.join(home, 'data', 'sample_df_timeline.csv')
users = pd.read_csv(users_data_path)
df_timeline = pd.read_csv(timeline_data_path)

# process_data
num_cols = users.select_dtypes(include=[np.number]).columns
cat_cols = users.select_dtypes(exclude=[np.number]).columns

# process_data
print(f'Total number of users: {len(users)}, \nTotal number of unique users: {users.id.nunique()}, \n \nRedundant users ids: \n{users.id.value_counts().loc[lambda x: x > 1]}')
print('-------------------- \n')
redundant_ids = users.id.value_counts().loc[lambda x: x > 1].index.tolist()
print(users[users['id'].isin(redundant_ids)][['id', 'name', 'screen_name', 'location', 'depression']])
users = users[~users['id'].isin(redundant_ids)]

# process_data
print(users.protected.unique(), users.contributors_enabled.unique(), users.is_translator.unique(), users.following.unique(), users.follow_request_sent.unique(), users.notifications.unique())
users = users.drop(['id_str', 'protected', 'contributors_enabled', 'is_translator', 'following', 'follow_request_sent', 'notifications'], axis=1)

# process_data
logging.basicConfig(level=logging.WARNING)

# process_data
color_cols = ['profile_background_color', 'profile_link_color', 'profile_sidebar_border_color', 'profile_sidebar_fill_color', 'profile_text_color']
users = normalize_hex_color_columns(users, color_cols)
users = expand_hex_color_columns(users, color_cols)
users = add_color_brightness_columns(users, color_cols)

# visualize_data
color_columns = ['profile_background_color_brightness', 'profile_link_color_brightness', 'profile_sidebar_border_color_brightness', 'profile_sidebar_fill_color_brightness', 'profile_text_color_brightness']
corr_matrix = users[color_columns].corr()
plt.figure(figsize=(6, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.3)
plt.title('Correlation Matrix of Color Features', fontsize=10)
plt.show()

# process_data
users = users.drop(['profile_background_color_r', 'profile_background_color_g', 'profile_background_color_b', 'profile_link_color_r', 'profile_link_color_g', 'profile_link_color_b', 'profile_sidebar_border_color_r', 'profile_sidebar_border_color_g', 'profile_sidebar_border_color_b', 'profile_sidebar_fill_color_r', 'profile_sidebar_fill_color_g', 'profile_sidebar_fill_color_b', 'profile_text_color_r', 'profile_text_color_g', 'profile_text_color_b', 'profile_link_color_brightness', 'profile_sidebar_border_color_brightness', 'profile_sidebar_fill_color_brightness', 'profile_text_color_brightness'], axis=1)

# process_data
users['created_at'] = pd.to_datetime(users['created_at'], errors='coerce', utc=True)
users['account_age'] = (pd.Timestamp.now(tz='UTC') - users['created_at']).dt.days / 365
print(f'The account_age min value (in years): {users.account_age.min():.2f}, \nThe account_age max value (in years): {users.account_age.max():.2f}')

# process_data
numeric_columns = users.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'id' in numeric_columns:
    numeric_columns.remove('id')
print(f'Numeric Columns: {numeric_columns}')
scaler = MinMaxScaler()
users[numeric_columns] = scaler.fit_transform(users[numeric_columns])

# process_data
bool_columns = users.select_dtypes(include=['bool']).columns.tolist()
print(f'boolean Columns: {bool_columns}')
users[bool_columns] = users[bool_columns].astype(int)

# process_data
users['creation_year'] = users['created_at'].dt.year

# visualize_data
numerical_features = ['followers_count', 'friends_count', 'listed_count', 'favourites_count', 'statuses_count', 'profile_background_color_brightness', 'account_age']
fig1 = plot_feature_bars(users, numerical_features, mode='numerical')
fig1.show()

# visualize_data
categorical_features = ['creation_year', 'geo_enabled', 'profile_use_background_image', 'has_extended_profile', 'default_profile', 'default_profile_image', 'verified', 'is_translation_enabled']
fig2 = plot_feature_bars(users, categorical_features, mode='categorical')
fig2.show()

# visualize_data
fig = plot_correlation_heatmap(df=users, method='pearson', show_values=False, mask_triangle='upper')
fig.show()

# process_data
users = users[['id', 'followers_count', 'friends_count', 'listed_count', 'favourites_count', 'statuses_count', 'geo_enabled', 'profile_use_background_image', 'has_extended_profile', 'default_profile', 'default_profile_image', 'profile_background_color_brightness', 'account_age', 'verified', 'is_translation_enabled', 'depression']]

# process_data
user_counts_pos = df_timeline[df_timeline['depression'] == True].groupby('user_id').size().reset_index(name='count')
user_counts_pos = user_counts_pos.sort_values(by='count', ascending=False)
user_counts_neg = df_timeline[df_timeline['depression'] == False].groupby('user_id').size().reset_index(name='count')
user_counts_neg = user_counts_neg.sort_values(by='count', ascending=False)

# process_data
df_timeline['created_at'] = df_timeline['created_at'].progress_apply(lambda x: x if isinstance(x, pd.Timestamp) else parser.parse(str(x)))
print(df_timeline['created_at'].apply(type).value_counts())
(df_timeline, df_source) = clean_source(df_timeline)
df_agg = compute_aggregates(df_timeline)
df_freq = compute_tweet_frequency(df_timeline)
df_quotes = compute_quote_features(df_timeline)
df_interactions = compute_interaction_features(df_timeline)
df_unique = df_timeline.drop_duplicates(subset=['user_id', 'text'])
df_final = df_agg.merge(df_freq, on='user_id', how='left').merge(df_interactions, on='user_id', how='left').merge(df_source, on='user_id', how='left').merge(df_quotes, on='user_id', how='left')
df_final = df_final.merge(df_timeline[['user_id', 'depression']].drop_duplicates(), on='user_id', how='left')
df_final.drop(columns=['total_timespan'], inplace=True)

# process_data
df_final['mean_time_gap'] = df_final.groupby('depression')['mean_time_gap'].transform(lambda x: x.fillna(x.mean()))
df_final['median_time_gap'] = df_final.groupby('depression')['median_time_gap'].transform(lambda x: x.fillna(x.mean()))
df_final['std_time_gap'] = df_final.groupby('depression')['std_time_gap'].transform(lambda x: x.fillna(x.mean()))
df_final['interactions_with_depressed'] = df_final['interactions_with_depressed'].fillna(0)
df_final['total_interactions'] = df_final['total_interactions'].fillna(0)
df_final['interactions_with_non_depressed'] = df_final['interactions_with_non_depressed'].fillna(0)
df_final = df_final.drop(columns=['quotes_from_depressed', 'total_quotes_y', 'quotes_from_non_depressed'])
"\ndf_final['quotes_from_depressed'] = df_final['quotes_from_depressed'].fillna(0)\ndf_final['total_quotes_y'] = df_final['total_quotes_y'].fillna(0)\ndf_final['quotes_from_non_depressed'] = df_final['quotes_from_non_depressed'].fillna(0)\n"

# process_data
num_cols = df_final.select_dtypes(include=[np.number]).columns
cat_cols = df_final.select_dtypes(exclude=[np.number]).columns

# process_data
df_final['depression'] = df_final['depression'].astype(int)
numerical_features = ['total_retweets', 'total_favorites', 'avg_retweets', 'avg_favorites', 'tweet_count', 'mean_time_gap', 'median_time_gap', 'std_time_gap', 'tweet_rate', 'unique_interactions', 'total_interactions', 'unique_sources', 'total_quotes_x', 'quote_ratio']
plot_feature_bars(df_final, numerical_features, mode='numerical')

# process_data
numeric_columns = df_final.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'depression' in numeric_columns:
    numeric_columns.remove('depression')
if 'user_id' in numeric_columns:
    numeric_columns.remove('user_id')
print(f'Numeric Columns: {numeric_columns}')
scaler = MinMaxScaler()
df_final[numeric_columns] = scaler.fit_transform(df_final[numeric_columns])
bool_columns = df_final.select_dtypes(include=['bool']).columns.tolist()
print(f'boolean Columns: {bool_columns}')
df_final[bool_columns] = df_final[bool_columns].astype(int)

# process_data
users = users[['id', 'followers_count', 'listed_count', 'favourites_count', 'statuses_count', 'has_extended_profile', 'default_profile', 'profile_background_color_brightness', 'account_age', 'depression']]

# process_data
df_final.rename(columns={'user_id': 'id'}, inplace=True)

# process_data
merged_df = pd.merge(df_final, users, on=['id', 'depression'], how='inner')

# process_data
class_distribution = merged_df['depression'].value_counts(normalize=True) * 100
class_distribution

# visualize_data
fig = plot_correlation_heatmap(df=merged_df, method='pearson', show_values=False, mask_triangle='upper')
fig.show()

# process_data
merged_df = merged_df[['id', 'avg_retweets', 'avg_favorites', 'tweet_count', 'mean_time_gap', 'tweet_rate', 'unique_interactions', 'interactions_with_depressed', 'total_interactions', 'unique_sources', 'quote_ratio', 'followers_count', 'favourites_count', 'statuses_count', 'has_extended_profile', 'profile_background_color_brightness', 'account_age', 'depression']]

# transfer_results
save_path = os.path.join(home, 'sample_structured_data.csv')
merged_df.to_csv(save_path, index=False)

# train_model
RANDOM_SEED = 42

# process_data
optimized_results = optimize_models(merged_df, n_trials=10)

# train_model
X = merged_df.drop(['id', 'depression'], axis=1)
y = merged_df['depression']
best_model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.22132944375008318, max_depth=5, subsample=0.7098092451187741, min_samples_split=3, min_samples_leaf=4, random_state=42)
best_model.fit(X, y)
feature_importance_values = best_model.feature_importances_
feature_names = merged_df.drop(columns=['id', 'depression']).columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_values})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
feature_importance

# train_model
(X_train, X_test, y_train, y_test) = prepare_data(merged_df)
perm = PermutationImportance(best_model, scoring='accuracy', random_state=42).fit(X_test, y_test)
eli5.show_weights(perm, feature_names=X_test.columns.tolist())

# evaluate_model
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(), class_names=['No Depression', 'Depression'], mode='classification')
exp = explainer.explain_instance(X_test.iloc[0].values, best_model.predict_proba)
exp.show_in_notebook()

# evaluate_model
exp = explainer.explain_instance(X_test.iloc[-1].values, best_model.predict_proba)
exp.show_in_notebook()
