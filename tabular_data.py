import pandas as pd
import ast


def remove_rows_with_missing_ratings(dataset: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """This function is used to remove null values from the rating columns:
        'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 
        'Location_rating', 'Check-in_rating', 'Value_rating'

    Args: 
        dataset (pd.core.frame.DataFrame): raw dataframe
    
    Returns:
        airbnb_dataset (pd.core.frame.DataFrame): dataset with rows removed
        which has null values in the rating columns.
    """

    airbnb_dataset = dataset.copy()
    ratings = list()
    columns = airbnb_dataset.columns
    for column in columns:
        if str(column).endswith("_rating"):
            ratings.append(column)
    airbnb_dataset[ratings].isna().sum()
    airbnb_dataset.dropna(subset=ratings, inplace=True)
    airbnb_dataset[ratings].isna().sum()

    return airbnb_dataset

def combine_description_strings(dataset: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """This function is used to: 
        parse the string into a list, remove empty quotes, remove any records with a missing 
        description and remove "About this space" prefix from every description 
    
    Args: 
        dataset (pd.core.frame.DataFrame): dataset with rows removed
        which has null values in the rating columns.
 
    Returns:
        dataset (pd.core.frame.DataFrame): processed dataset with combined string in the 
        "Description" column
    """
    
    dataset.dropna(subset='Description', inplace=True)
    
    def __process_description(description: str) -> str:
        try:
            texts = ast.literal_eval(description)
            processed_description = list()
            for text in texts:
                if text == "" or text == "About this space":
                    continue
                processed_description.append(text)

            return " ".join(processed_description)
        except:
            return description
        
    dataset.Description = dataset.Description.apply(__process_description)
    
    return dataset

def set_default_feature_values(dataset: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """This function is used to replace null values with "1" in the column: 
        'guests', 'beds', 'bathrooms', 'bedrooms' 

    Args:
        dataset (pd.core.frame.DataFrame): processed dataset with combined string in the 
        "Description" column

    Returns:
        dataset (pd.core.frame.DataFrame): processed dataset with feature value set to "1" 
        in the place of null values in 4 columns: 'guests', 'beds', 'bathrooms', 'bedrooms'
    """

    rows = ['guests', 'beds', 'bathrooms', 'bedrooms']
    for row in rows:        
        dataset[row].fillna("1", inplace=True)
       
    return dataset

def clean_tabular_data(dataset: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """This function will clean the dataset:
        -> Call remove_rows_with_missing_ratings(): remove rows having null values in rating columns
        -> Call combine_description_strings(): remove empty spaces and combine the string in description column
        -> Call set_default_feature_values(): set the null values to "1" in the feature columns

    Args: 
        dataset (pd.core.frame.DataFrame): raw dataframe

    Returns:
        dataset (pd.core.frame.DataFrame): cleaned dataset 
    """

    dataset = remove_rows_with_missing_ratings(dataset)
    dataset = combine_description_strings(dataset)
    dataset = set_default_feature_values(dataset)

    return dataset

if __name__ == "__main__":
    dataset = pd.read_csv("airbnb-property-listings/tabular_data/listing.csv")
    cleaned_dataset = clean_tabular_data(dataset)
    cleaned_dataset.to_csv("clean_tabular_data.csv", index=None)