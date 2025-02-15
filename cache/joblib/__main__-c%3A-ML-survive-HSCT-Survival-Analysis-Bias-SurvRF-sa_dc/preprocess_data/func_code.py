# first line: 47
@memory.cache
def preprocess_data(train_data):
    # Identify numerical and categorical features
    numerical_features = train_data.select_dtypes(include=['float32', 'float64']).columns
    categorical_features = train_data.select_dtypes(include=['category']).columns
    
    # Convert categorical to string type for encoding
    for cat_feature in categorical_features:
        train_data[cat_feature] = train_data[cat_feature].astype(str)
    
    # Impute missing values
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    
    # Impute and convert back to efficient dtypes
    train_data[numerical_features] = num_imputer.fit_transform(train_data[numerical_features]).astype('float32')
    train_data[categorical_features] = cat_imputer.fit_transform(train_data[categorical_features])
    
    # Encode categorical features
    ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_data[categorical_features] = ordinal_encoder.fit_transform(train_data[categorical_features])
    
    return train_data, numerical_features, categorical_features
