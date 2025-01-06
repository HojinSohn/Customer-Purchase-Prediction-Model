import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score

def get_seed():
    # Set the seed for reproducibility, you can change this to any number
    return 73

def load_data_types(basepath: Path):
    # Some of the data types are going to be objects, we need to change them based on the data_type_dict json file. 
    # For the source and target entity do not include thier primary keys (costumer_id and product_id), beacuase
    # they will not be used as input features. They should be in a seperate variable
    with open(basepath / "data_type_dict.json", 'r') as file:
        dataTypeDic = json.load(file)

    src_type = dataTypeDic['src_entity']
    tar_type = dataTypeDic['tar_entity']

    src_entity = list(dataTypeDic['pkeys'].keys())[0]
    tar_entity = list(dataTypeDic['pkeys'].keys())[1]

    return src_type, tar_type, src_entity, tar_entity

def load_data(basepath: Path, src_type, tar_type, train=True):
    # Since we want to predict the list of product IDs a customer (that is identify with a customer ID) will buy, the source
    # entity is the customer and the target entity is the product. The reationship between them is the transactions table. 
    # Load the source and target entity tables, and force the correct data types
    source_entity_table = pd.read_csv(basepath / "customers.csv").astype(src_type)
    target_entity_table = pd.read_csv(basepath / "products.csv").astype(tar_type)

    data = []

    if train:
        # Load the data for training, validation
        train_data = pd.read_csv(basepath / "train_data.csv")
        val_data = pd.read_csv(basepath / "validation_data.csv")  

        # Replace spaces with comas in the list_product_id column so we can have a list of product_ids
        train_data['list_product_id'] = train_data['list_product_id'].apply(lambda text: text.replace(' ', ', '))
        val_data['list_product_id'] = val_data['list_product_id'].apply(lambda text: text.replace(' ', ', '))

        # Rename the columns of the train and valid data to match the target entity 
        train_data = train_data.rename(columns={'list_product_id': 'product_id'})
        val_data = val_data.rename(columns={'list_product_id': 'product_id'})

        data += [train_data, val_data]
    else:
        #Load the test data
        customer_id_df = pd.read_csv(basepath / "to_predict.csv")
        data += [customer_id_df]

    return data + [source_entity_table, target_entity_table]

def update_data(source_entity_table,target_entity_table,new_features, src_entity, src_type:dict, tar_entity, tar_type:dict):
    # Update the source and target entity tables with the new features
    source_entity_table = source_entity_table.merge(new_features[0], how="left", on=src_entity)
    target_entity_table = target_entity_table.merge(new_features[1], how="left", on=tar_entity)
    # Update the data types of the source and target entity tables with the new features
    src_type.update(new_features[2])
    tar_type.update(new_features[3])
    # Combine all data types
    all_types = src_type
    all_types.update(tar_type)

    return source_entity_table, target_entity_table, all_types

def negative_sampling(df: pd.DataFrame, tar_entity: str, tar_entity_df: pd.DataFrame, target_col_name: str) -> pd.DataFrame:
    # Create a negative sampling df, containing source and target entities pairs,
    # such that there are no links between them.
    negative_sample_df_columns = list(df.columns)
    negative_sample_df_columns.remove(tar_entity)
    negative_samples_df = df[negative_sample_df_columns]
    negative_samples_df[tar_entity] = np.random.choice(
        tar_entity_df[tar_entity], size=len(negative_samples_df)
    )
    negative_samples_df[target_col_name] = 0

    return negative_samples_df

def pair_feature_engineering(df: pd.DataFrame, all_types: dict, customers_features=None, products_features=None):
    print(df.columns.tolist())
    # add src-tar pair match columns 
    df['perceived_color_match'] = ((~df['top_perceived_color'].isna()) & (~df['perceived_colour_value_id'].isna()) & (df['top_perceived_color'] == df['perceived_colour_value_id']))
    
    df['prod_group_match'] = ((~df['top_prod_group'].isna()) & (~df['prod_group_name'].isna()) & (df['top_prod_group'] == df['prod_group_name']))
                                   
    df['idx_group_match'] = ((~df['top_idx_group_no'].isna()) & (~df['idx_group_no'].isna()) & (df['top_idx_group_no'] == df['idx_group_no']))
    
           
    df['sec_no_match'] = ((~df['top_sec_no'].isna()) & (~df['sec_no'].isna()) & (df['top_sec_no'] == df['sec_no']))
           
    df['garment_group_no_match'] = ((~df['top_garment_group_no'].isna()) & (~df['garment_group_no'].isna()) & (df['top_garment_group_no'] == df['garment_group_no']))
           
    df['idx_code_match'] = ((~df['top_idx_code'].isna()) & (~df['idx_code'].isna()) & (df['top_idx_code'] == df['idx_code']))
           
    df['department_no_match'] = ((~df['top_department_no'].isna()) & (~df['department_no'].isna()) & (df['top_department_no'] == df['department_no']))
           
    df['prod_type_no_match'] = ((~df['top_prod_type_no'].isna()) & (~df['prod_type_no'].isna()) & (df['top_prod_type_no'] == df['prod_type_no']))
           
    df['graph_appearance_no_match'] = ((~df['top_graph_appearance_no'].isna()) & (~df['graph_appearance_no'].isna()) & (df['top_graph_appearance_no'] == df['graph_appearance_no']))
    
   
    df['colour_group_code_match'] = ((~df['top_colour_group_code'].isna()) & (~df['colour_group_code'].isna()) & (df['top_colour_group_code'] == df['colour_group_code']))
           
    df['perceived_colour_master_id_match'] = ((~df['top_perceived_colour_master_id'].isna()) & (~df['perceived_colour_master_id'].isna()) & (df['top_perceived_colour_master_id'] == df['perceived_colour_master_id']))
 

    df['age_similarity'] = ((df['age'] - df['average_age']) / df['std_age']).where(~df['age'].isna() & ~df['average_age'].isna() & ~df['std_age'].isna(), np.nan)
    df['FN_similarity'] = ((df['FN'] - df['average_FN']) / df['std_FN']).where(~df['FN'].isna() & ~df['average_FN'].isna() & ~df['std_FN'].isna(), np.nan)
    df['active_similarity'] = ((df['active_status'] - df['average_active']) / df['std_active']).where(~df['active_status'].isna() & ~df['average_active'].isna() & ~df['std_active'].isna(), np.nan)
    df['member_similarity'] = ((df['member_status'] - df['average_member']) / df['std_member']).where(~df['member_status'].isna() & ~df['average_member'].isna() & ~df['std_member'].isna(), np.nan)
    df['news_similarity'] = ((df['news_frequency'] - df['average_news']) / df['std_news']).where(~df['news_frequency'].isna() & ~df['average_news'].isna() & ~df['std_news'].isna(), np.nan)

    
    

    all_types.update({'perceived_color_match': 'bool', 'prod_group_match': 'bool', 'idx_group_match': 'bool', 'sec_no_match': 'bool', 'garment_group_no_match': 'bool', 'idx_code_match': 'bool', 'department_no_match': 'bool', 'prod_type_no_match': 'bool', 'graph_appearance_no_match': 'bool', 'colour_group_code_match': 'bool', 'perceived_colour_master_id_match': 'bool', 'age_similarity': 'float64', 'FN_similarity': 'float64', 'active_similarity': 'float64', 'member_similarity': 'float64', 'news_similarity': 'float64'})
    df = df.astype(all_types)
    
    # remove "top"columns from table after adding src-tar match columns
    columns_to_drop = {'top_perceived_color', 'top_prod_group', 'top_idx_group_no', 'top_sec_no', 'top_garment_group_no', 'top_idx_code', 'top_department_no', 'top_prod_type_no', 'top_graph_appearance_no', 'top_colour_group_code', 'top_perceived_colour_master_id', 'postal_code', 'product_code', 'prod_name', 'prod_type_name', 'graph_appearance_name', 'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name', 'idx_name', 'idx_group_name', 'sec_name', 'garment_group_name', 'detail_desc', 'prod_type_no', 'graph_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'idx_code', 'idx_group_no', 'sec_no', 'garment_group_no', 'FN', 'active_status', 'member_status', 'news_frequency', 'age', 'average_age', 'std_age', 'average_FN', 'average_active', 'average_member', 'average_news', 'std_FN', 'std_active', 'std_member', 'std_news'}
    
    
    
    
    df.drop(columns = columns_to_drop, inplace = True)
    
    # remove those fields from customers_features and products_features table
    if customers_features is not None:
        customers_features = customers_features.drop(columns = {'top_perceived_color', 'top_prod_group', 'top_idx_group_no', 'top_sec_no', 'top_garment_group_no', 'top_idx_code', 'top_department_no', 'top_prod_type_no', 'top_graph_appearance_no', 'top_colour_group_code', 'top_perceived_colour_master_id', 'postal_code', 'FN', 'active_status', 'member_status', 'news_frequency', 'age'})
        
#         'FN', 'active_status', 'member_status', 'news_frequency', 'age'
        
    if products_features is not None:
        products_features = products_features.drop(columns = {'product_code', 'prod_name', 'prod_type_name', 'graph_appearance_name', 'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name', 'idx_name', 'idx_group_name', 'sec_name', 'garment_group_name', 'detail_desc', 'prod_type_no', 'graph_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'idx_code', 'idx_group_no', 'sec_no', 'garment_group_no', 'average_age', 'std_age', 'average_FN', 'average_active', 'average_member', 'average_news', 'std_FN', 'std_active', 'std_member', 'std_news'}) 
        
#         'prod_type_no', 'graph_appearance_no', 'color_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'idx_code', 'idx_group_no', 'sec_no', 'garment_group_no', 'average_age', 'std_age', 'average_FN', 'average_active', 'average_member', 'average_news'
        
        
        
    all_types_filter = {key: value for key, value in all_types.items() if key not in columns_to_drop}

    pair_features_columns = ['perceived_color_match', 'prod_group_match', 'idx_group_match', 'sec_no_match', 'garment_group_no_match', 'idx_code_match', 'department_no_match', 'prod_type_no_match', 'graph_appearance_no_match', 'colour_group_code_match', 'perceived_colour_master_id_match', 'age_similarity', 'FN_similarity', 'active_similarity', 'member_similarity', 'news_similarity']
    
    
#         #     #remove all the features irrelevant to pair 4
#     columns_to_drop = {'postal_code', 'product_code', 'prod_name', 'prod_type_name', 'graph_appearance_name', 'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name', 'idx_name', 'idx_group_name', 'sec_name', 'garment_group_name', 'detail_desc'}
    
#     df.drop(columns = columns_to_drop, inplace = True)
    
    
    # new customer products features with columns dropped / original kept same
    return df, all_types_filter, customers_features, products_features, df[pair_features_columns]
            
            
# Change this function to include your new features and make your data for training is more complex
def feature_engineering(basepath: Path, src_entity: str, tar_entity: str, source_entity_table: pd.DataFrame, target_entity_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    #Perform feature engineering on transaction data to extract customer and product features. You can create new features based on the existing ones, 
    # they dont need to be the same as the ones in the example below.
    
    # Load transaction data
    transaction_data = pd.read_csv(basepath / "transactions.csv")
    # Ensure transaction date is in datetime format
    transaction_data['time_date'] = pd.to_datetime(transaction_data['time_date'])
    
    
    source_entity_table['FN'] = source_entity_table['FN'].astype(float).fillna(-1)
    source_entity_table['active_status'] = source_entity_table['active_status'].astype(float).fillna(-1)
    
    member_status_mapping = {'ACTIVE': 1, 'PRE-CREATE': 1, 'LEFT CLUB': -1}
    source_entity_table['member_status'] = source_entity_table['member_status'].map(member_status_mapping).astype(float).fillna(0)
    
    news_frequency_mapping = {'NONE': -1, 'Regularly': 1, 'Monthly': 0.5}
    source_entity_table['news_frequency'] = source_entity_table['news_frequency'].map(news_frequency_mapping).astype(float).fillna(0)
    
    transaction_data_expand = transaction_data.merge(source_entity_table, how="left", on=src_entity).merge(target_entity_table, how="left", on=tar_entity)
    
    print(transaction_data_expand.columns)
    
    
    
    # produce top perceived color column for each customer
    color_preference = (transaction_data_expand.groupby([src_entity, "perceived_colour_value_id"]).size().reset_index(name='color_count'))
    top_color_preference = (
        color_preference.sort_values([src_entity, 'color_count'], ascending=[True, False])
        .drop_duplicates(subset=[src_entity])  # Keeps only the top color per customer
        .rename(columns={'perceived_colour_value_id': 'top_perceived_color'})
        .drop('color_count', axis=1)
    )
    
    # produce top prod group column for each customer
    prod_group_preference = (transaction_data_expand.groupby([src_entity, "prod_group_name"]).size().reset_index(name='prod_count'))
    top_prod_group_preference = (
        prod_group_preference.sort_values([src_entity, 'prod_count'], ascending=[True, False])
        .drop_duplicates(subset=[src_entity])  # Keeps only the top prod_group per customer
        .rename(columns={'prod_group_name': 'top_prod_group'})
        .drop('prod_count', axis=1)
    )
    
    # produce top idx group preference for each customer
    idx_group_preference = (transaction_data_expand.groupby([src_entity, "idx_group_no"]).size().reset_index(name='idx_group_no_count'))
    top_idx_group_preference = (
        idx_group_preference.sort_values([src_entity, 'idx_group_no_count'], ascending=[True, False])
        .drop_duplicates(subset=[src_entity])  # Keeps only the top prod_group per customer
        .rename(columns={'idx_group_no': 'top_idx_group_no'})
        .drop('idx_group_no_count', axis=1)
    )
    
    
    
    # produce top sec no for each customer
    sec_number_preference = (transaction_data_expand.groupby([src_entity, "sec_no"]).size().reset_index(name='sec_no_count'))
    top_sec_number_preference = (
        sec_number_preference.sort_values([src_entity, 'sec_no_count'], ascending=[True, False])
        .drop_duplicates(subset=[src_entity])  # Keeps only the top prod_group per customer
        .rename(columns={'sec_no': 'top_sec_no'})
        .drop('sec_no_count', axis=1)
    )
    
    # produce top garment group no for each customer
    garment_group_preference = (transaction_data_expand.groupby([src_entity, "garment_group_no"]).size().reset_index(name='garment_group_no_count'))
    top_garment_group_preference = (
        garment_group_preference.sort_values([src_entity, 'garment_group_no_count'], ascending=[True, False])
        .drop_duplicates(subset=[src_entity])  # Keeps only the top prod_group per customer
        .rename(columns={'garment_group_no': 'top_garment_group_no'})
        .drop('garment_group_no_count', axis=1)
    )
    
    # produce top idx code for each customer
    idx_code_preference = (transaction_data_expand.groupby([src_entity, "idx_code"]).size().reset_index(name='idx_code_count'))
    top_idx_code_preference = (
        idx_code_preference.sort_values([src_entity, 'idx_code_count'], ascending=[True, False])
        .drop_duplicates(subset=[src_entity])  # Keeps only the top prod_group per customer
        .rename(columns={'idx_code': 'top_idx_code'})
        .drop('idx_code_count', axis=1)
    )
    
    # produce top department number for each customer
    department_preference = (transaction_data_expand.groupby([src_entity, "department_no"]).size().reset_index(name='department_no_count'))
    top_department_preference = (
        department_preference.sort_values([src_entity, 'department_no_count'], ascending=[True, False])
        .drop_duplicates(subset=[src_entity])  # Keeps only the top prod_group per customer
        .rename(columns={'department_no': 'top_department_no'})
        .drop('department_no_count', axis=1)
    )
    
    # produce top prod type number for each customer
    prod_type_preference = (transaction_data_expand.groupby([src_entity, "prod_type_no"]).size().reset_index(name='prod_type_no_count'))
    top_prod_type_preference = (
        prod_type_preference.sort_values([src_entity, 'prod_type_no_count'], ascending=[True, False])
        .drop_duplicates(subset=[src_entity])  # Keeps only the top prod_group per customer
        .rename(columns={'prod_type_no': 'top_prod_type_no'})
        .drop('prod_type_no_count', axis=1)
    )
    
    # produce top graph appearance number for each customer
    graph_appearance_preference = (transaction_data_expand.groupby([src_entity, "graph_appearance_no"]).size().reset_index(name='graph_appearance_no_count'))
    top_graph_appearance_preference = (
        graph_appearance_preference.sort_values([src_entity, 'graph_appearance_no_count'], ascending=[True, False])
        .drop_duplicates(subset=[src_entity])  # Keeps only the top prod_group per customer
        .rename(columns={'graph_appearance_no': 'top_graph_appearance_no'})
        .drop('graph_appearance_no_count', axis=1)
    )
    
    # produce top colour group code for each customer
    colour_group_code_preference = (transaction_data_expand.groupby([src_entity, "colour_group_code"]).size().reset_index(name='colour_group_code_count'))
    top_colour_group_code_preference = (
        colour_group_code_preference.sort_values([src_entity, 'colour_group_code_count'], ascending=[True, False])
        .drop_duplicates(subset=[src_entity])  # Keeps only the top prod_group per customer
        .rename(columns={'colour_group_code': 'top_colour_group_code'})
        .drop('colour_group_code_count', axis=1)
    )
    
    
    
    # produce top perceived colour master id for each customer
    perceived_colour_master_preference = (transaction_data_expand.groupby([src_entity, "perceived_colour_master_id"]).size().reset_index(name='perceived_colour_master_id_count'))
    top_perceived_colour_master_preference = (
        perceived_colour_master_preference.sort_values([src_entity, 'perceived_colour_master_id_count'], ascending=[True, False])
        .drop_duplicates(subset=[src_entity])  # Keeps only the top prod_group per customer
        .rename(columns={'perceived_colour_master_id': 'top_perceived_colour_master_id'})
        .drop('perceived_colour_master_id_count', axis=1)
    )
    
    
    
    
    
    
    
    # Create customer-related features
    customer_new_features = transaction_data.groupby(src_entity).agg({
        # Recency, first purchase date
        'time_date': ['max', 'min'],
        # Purchase frequency  
        src_entity: 'count',
        # Monetary value           
        'price': 'sum'
        # TODO get the perceived color preference
        # TODO get prod_group preference
        # TODO get idx_group preference
    }).reset_index()
    
    # Flatten multi-level columns
    customer_new_features.columns = [src_entity, 'last_purchase_date', 'first_purchase_date', 'total_purchases', 'total_spent']
    
    # Add "top" columns into customer_features table
    customer_new_features = customer_new_features.merge(top_color_preference, how='left', on=src_entity)
    customer_new_features = customer_new_features.merge(top_prod_group_preference, how='left', on=src_entity)
    customer_new_features = customer_new_features.merge(top_idx_group_preference, how='left', on=src_entity)
    
    customer_new_features = customer_new_features.merge(top_sec_number_preference, how='left', on=src_entity)
    customer_new_features = customer_new_features.merge(top_garment_group_preference, how='left', on=src_entity)
    customer_new_features = customer_new_features.merge(top_idx_code_preference, how='left', on=src_entity)
    customer_new_features = customer_new_features.merge(top_department_preference, how='left', on=src_entity)
    customer_new_features = customer_new_features.merge(top_prod_type_preference, how='left', on=src_entity)
    customer_new_features = customer_new_features.merge(top_graph_appearance_preference, how='left', on=src_entity)
    customer_new_features = customer_new_features.merge(top_colour_group_code_preference, how='left', on=src_entity)
    customer_new_features = customer_new_features.merge(top_perceived_colour_master_preference, how='left', on=src_entity)
   
    
    # Flatten multi-level columns
    customer_new_features.columns = [src_entity, 'last_purchase_date', 'first_purchase_date', 'total_purchases', 'total_spent', 'top_perceived_color', 'top_prod_group', 'top_idx_group_no', 'top_sec_no', 'top_garment_group_no', 'top_idx_code', 'top_department_no', 'top_prod_type_no', 'top_graph_appearance_no', 'top_colour_group_code', 'top_perceived_colour_master_id']
    # Create a dictionary to specify the data types of the new features
    customer_new_features_type = {'last_purchase_date': 'category', 'first_purchase_date': 'category', 'total_purchases': 'Int64', 'total_spent': 'float64', 'top_perceived_color': 'category', 'top_prod_group': 'category', 'top_idx_group_no': 'category', 'top_sec_no': 'category', 'top_garment_group_no': 'category', 'top_idx_code': 'category', 'top_department_no': 'category', 'top_prod_type_no': 'category', 'top_graph_appearance_no': 'category', 'top_colour_group_code': 'category', 'top_perceived_colour_master_id': 'category'}
    customer_new_features.astype(customer_new_features_type)

    
    
    # Create product-related features
    product_pair_features = transaction_data_expand.groupby(tar_entity).agg({
        # Average age, variation
        'age': ['mean', 'std'], 
        'FN': ['mean', 'std'], 
        'active_status': ['mean', 'std'], 
        'member_status': ['mean', 'std'], 
        'news_frequency': ['mean', 'std'],    
    }).reset_index()
    # Flatten multi-level columns
    product_pair_features.columns = [tar_entity, 'average_age', 'std_age', 'average_FN', 'std_FN', 'average_active', 'std_active', 'average_member', 'std_member', 'average_news', 'std_news']
    product_pair_features['std_age'] = product_pair_features['std_age'].replace(0, 1)
    product_pair_features['std_FN'] = product_pair_features['std_FN'].replace(0, 1)
    product_pair_features['std_active'] = product_pair_features['std_active'].replace(0, 1)
    product_pair_features['std_member'] = product_pair_features['std_member'].replace(0, 1)
    product_pair_features['std_news'] = product_pair_features['std_news'].replace(0, 1)
    
    
    # Create product-related features
    product_new_features = transaction_data.groupby(tar_entity).agg({
        # Average price, price variation
        'price': ['mean', 'std'],
        # Product popularity  
        tar_entity: 'count'         
    }).reset_index()
    
    # Flatten multi-level columns
    product_new_features.columns = [tar_entity, 'average_price', 'price_std', 'popularity']
    
    
    product_new_features = product_new_features.merge(product_pair_features, how='left', on=tar_entity)
    
    product_new_features.columns = [tar_entity, 'average_price', 'price_std', 'popularity', 'average_age', 'std_age', 'average_FN', 'std_FN', 'average_active', 'std_active', 'average_member', 'std_member', 'average_news', 'std_news']
    
    # Create a dictionary to specify the data types of the new features
    product_new_features_type = {'average_price': 'float64', 'price_std': 'float64', 'popularity': 'Int64', 'average_age': 'float64', 'std_age': 'float64', 'average_FN': 'float64', 'std_FN': 'float64', 'average_active': 'float64', 'std_active': 'float64', 'average_member': 'float64', 'std_member': 'float64', 'average_news': 'float64', 'std_news': 'float64'}
    product_new_features.astype(product_new_features_type)

    return customer_new_features, product_new_features, customer_new_features_type, product_new_features_type

def get_split_train_val(dfs, target_col_name: str):
    # Split the features (X) and target (y) from train and validation and drop the target column from the features
    X_train = dfs["train"].drop(columns=[target_col_name])
    y_train = dfs["train"][target_col_name]

    X_val = dfs["val"].drop(columns=[target_col_name])
    y_val = dfs["val"][target_col_name]

    return X_train, y_train, X_val, y_val

def evaluate(xgb_model, X, y, typ: str):
    # Evaluate the model
    # Predict for the train and validation data
    y_pred = xgb_model.predict_proba(X)

    # Create a mask to convert the probabilities to binary values
    mask = (y_pred[:,1] >= 0.5).astype(int)

    # Calculate model metrics
    accuracy = accuracy_score(y, mask)
    precision = precision_score(y, mask)
    recall = recall_score(y, mask)
    map_T = np.mean(average_precision_score(y, mask))

    # Print the results
    print(f"{typ} Accuracy: {accuracy}")
    print(f"{typ} Presicion: {precision}")
    print(f"{typ} Recall: {recall}")
    print(f"{typ} Mean Average Precision: {map_T}\n")

def create_submission_file(basepath: Path):
    filepath = basepath / 'predictions.txt' 
    output_path = basepath / 'submission_file.csv'

    # Read the .txt file
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Initialize an empty list to store processed data
    data = []

    # Iterate through each line in the file, skipping the header if present
    for line in lines[1:]:
        # Extract the customer_id and the list of product_id from each line
        customer_id, product_list = line.strip().split(",", 1)
        # Use regular expressions to extract the product IDs (keeping the original list structure)
        product_ids = re.findall(r'\d+', product_list)
        # Join product IDs as a single space-separated string and store the result
        data.append([int(customer_id), '[' + ' '.join(product_ids) + ']'])  # maintain bracket format

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['customer_id', 'list_product_id'])

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False) 