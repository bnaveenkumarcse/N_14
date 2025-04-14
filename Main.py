import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_column_features(df, dataset_name="UploadedFile"):
    """
    Extracts key metadata and heuristics from each column in the DataFrame.

    Args:
        df (pd.DataFrame): The dataset to analyze.
        dataset_name (str): Optional name of the dataset.

    Returns:
        pd.DataFrame: A DataFrame containing feature metadata for each column.
    
    """
    feature_list = []
    for col in df.columns:
        col_data = df[col]
        features = {
            'dataset': dataset_name,
            'column_name': col,
            'dtype': str(col_data.dtype),
            'sample_value': str(col_data.iloc[0]),
            'has_keyword_quantity': int('quantity' in col.lower() or 'count' in col.lower() or 'item' in col.lower()),
            'has_keyword_sales': int('amount' in col.lower() or 'price' in col.lower() or 'total' in col.lower() or 'value' in col.lower() or 'sales' in col.lower()),
            'has_keyword_product': int('product' in col.lower() or 'item' in col.lower() or 'category' in col.lower() or 'name' in col.lower()),
            'has_keyword_date': int(any(x in col.lower() for x in ['date', 'order', 'invoice', 'sale', 'placed'])),
            'has_keyword_gender': int('gender' in col.lower() or 'sex' in col.lower()),
            'is_numeric': pd.api.types.is_numeric_dtype(col_data),
            'mean_value': col_data.mean() if pd.api.types.is_numeric_dtype(col_data) else None,
            'unique_values': col_data.nunique(),
            'label': 'quantity' if col.lower() in ['quantity', 'items', 'count', 'number_of_items'] else 'not_quantity'
        }
        feature_list.append(features)
    return pd.DataFrame(feature_list)

def read_file(file_path):
    """
    Reads a CSV file and loads it into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.

    """
    data = pd.read_csv(file_path)
    print("✅ File uploaded successfully!")
    return data

def get_columns_names(data):
    """
    Returns a string listing all column names in the DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame.

    Returns:
        str: A message listing column names.
    """
    col_names = data.columns
    return f'This file contains the following column names: {col_names}'

def calculate_average_basket_size(df: pd.DataFrame, quantity_col: str):
    """
    Calculates the average basket size based on a quantity column.

    Args:
        df (pd.DataFrame): The dataset.
        quantity_col (str): The column name representing quantity/items.

    Returns:
        tuple: ('Avg_Basket_Size', value)
    """


    total_items = df[quantity_col].sum()
    total_transactions = len(df)
    avg_basket_size = total_items / total_transactions
    return "Avg_Basket_Size", round(avg_basket_size, 2)

def calculate_average_basket_value(df: pd.DataFrame, sales_col: str):
    """
    Calculates the average basket value based on a sales column.

    Args:
        df (pd.DataFrame): The dataset.
        sales_col (str): The column name representing sales amount.

    Returns:
        tuple: ('Avg_Basket_Value', value)
    """
    total_sales = df[sales_col].sum()
    total_transactions = len(df)
    avg_basket_value = total_sales / total_transactions
    return "Avg_Basket_Value", round(avg_basket_value, 2)

def most_frequent_product_by_gender(df: pd.DataFrame, gender_col: str, product_col: str, quantity_col: str = None):
    """
    Analyzes and visualizes the most frequently purchased products by gender.

    Args:
        df (pd.DataFrame): The dataset.
        gender_col (str): The column representing gender.
        product_col (str): The product or category column.
        quantity_col (str, optional): Column for quantity. Uses counts if not provided.

    Returns:
        None
    """
    result = {}
    for gender in df[gender_col].dropna().unique():
        gender_df = df[df[gender_col] == gender]
        if quantity_col and quantity_col in df.columns:
            product_counts = gender_df.groupby(product_col)[quantity_col].sum()
        else:
            product_counts = gender_df[product_col].value_counts()
        
        most_frequent_product = product_counts.idxmax()
        count = product_counts.max()
        result[gender] = (most_frequent_product, int(count))
    for gender, (product, count) in result.items():
        print(f"{gender}: {product} (purchased {count} times)")
    genders = list(result.keys())
    products = [result[gender][0] for gender in genders]
    counts = [result[gender][1] for gender in genders]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=product_counts.values, y=product_counts.index, palette='viridis')

    plt.title(f'Products Purchased by {gender}')
    plt.xlabel('Purchase Count' if not quantity_col else f'Total {quantity_col}')
    plt.ylabel('Product')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genders, y=counts, hue=products, dodge=False, palette='Set2')

    plt.title("Most Frequently Purchased Products by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Purchase Count")
    plt.legend(title="Product", loc='upper right')
    plt.tight_layout()
    plt.show()
    print("\nMost Frequently Purchased Products by Gender:")
    for gender, (product, count) in result.items():
        print(f"{gender}: {product} (purchased {count} times)")

def get_valid_date_column(df, candidate_cols):
    """
    Tries to identify a valid date column from a list of candidates.

    Args:
        df (pd.DataFrame): The dataset.
        candidate_cols (list): List of possible date column names.

    Returns:
        str or None: Name of a valid date column, or None if not found.
    """
    for col in candidate_cols:
        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().sum() > 0:
                return col
        except:
            continue
    return None

def analyze_sales_trend(df, date_col, sales_col):
    """
    Analyzes and visualizes monthly sales trends and highlights the peak month.

    Args:
        df (pd.DataFrame): The dataset.
        date_col (str): Date column.
        sales_col (str): Sales value column.

    Returns:
        None
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['YearMonth'] = df[date_col].dt.to_period('M').astype(str)

    monthly_sales = df.groupby('YearMonth')[sales_col].sum().reset_index()
    peak_row = monthly_sales.loc[monthly_sales[sales_col].idxmax()]
    peak_month = peak_row['YearMonth']
    peak_value = round(peak_row[sales_col], 2)

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales['YearMonth'], monthly_sales[sales_col], marker='o', color='blue', linewidth=2)
    plt.axvline(peak_month, color='red', linestyle='--', label=f'🔝 Peak: {peak_month} ({peak_value})')
    plt.title("📊 Monthly Sales Trend", fontsize=16)
    plt.xlabel("Year-Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\n✅ Peak Sales Period: {peak_month} with sales value: {peak_value}")

def monthly_sales_by_category(df, date_col, sales_col, category_col):
    """
    Plots monthly sales trends broken down by category and prints peak months.

    Args:
        df (pd.DataFrame): The dataset.
        date_col (str): Date column.
        sales_col (str): Sales column.
        category_col (str): Product/category column.

    Returns:
        tuple: (Pivot table, DataFrame of peak sales by category)
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['YearMonth'] = df[date_col].dt.to_period('M').astype(str)

    grouped = df.groupby(['YearMonth', category_col])[sales_col].sum().reset_index()
    pivot_df = grouped.pivot(index='YearMonth', columns=category_col, values=sales_col)

    plt.figure(figsize=(14, 6))
    pivot_df.plot(marker='o')
    plt.title("📊 Monthly Sales Trend by Product Category", fontsize=16)
    plt.xlabel("Year-Month")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    peak_months = grouped.loc[grouped.groupby(category_col)[sales_col].idxmax()]
    print("\n🔝 Highest Sales Month for Each Category:")
    print(peak_months)
    return pivot_df, peak_months

def sales_by_year_and_month(df, date_col, sales_col):
    """
    Displays and visualizes total sales by year and by month (across years).

    Args:
        df (pd.DataFrame): The dataset.
        date_col (str): Date column.
        sales_col (str): Sales column.

    Returns:
        tuple: (DataFrame for yearly sales, DataFrame for monthly sales)
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month_name()

    sales_by_year = df.groupby('Year')[sales_col].sum().reset_index()
    print("\n📆 Total Sales by Year:")
    print(sales_by_year)

    sales_by_month = df.groupby('Month')[sales_col].sum().reset_index()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    sales_by_month['Month'] = pd.Categorical(sales_by_month['Month'], categories=month_order, ordered=True)
    sales_by_month = sales_by_month.sort_values('Month')

    print("\n📅 Total Sales by Month (across years):")
    print(sales_by_month)

    plt.figure(figsize=(10, 5))
    plt.bar(sales_by_year['Year'].astype(str), sales_by_year[sales_col], color='skyblue')
    plt.title("📈 Total Sales by Year", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("Total Sales")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(sales_by_month['Month'], sales_by_month[sales_col], marker='o', color='green')
    plt.title("📅 Total Sales by Month (Across All Years)", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return sales_by_year, sales_by_month

def highest_sales_period(df, date_col, sales_col):
    """
    Identifies and plots the highest sales period (year and month).

    Args:
        df (pd.DataFrame): The dataset.
        date_col (str): Date column.
        sales_col (str): Sales column.

    Returns:
        tuple: (peak year, peak month, peak sales value)
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month_name()

    sales_by_month = df.groupby(['Year', 'Month'])[sales_col].sum().reset_index()

    peak_row = sales_by_month.loc[sales_by_month[sales_col].idxmax()]
    peak_year = peak_row['Year']
    peak_month = peak_row['Month']
    peak_sales = peak_row[sales_col]

    print(f"\n📊 Highest Sales Recorded: {peak_month} {peak_year} with total sales: {peak_sales}")

    plt.figure(figsize=(12, 6))
    sales_by_month['YearMonth'] = sales_by_month['Year'].astype(str) + '-' + sales_by_month['Month']
    plt.plot(sales_by_month['YearMonth'], sales_by_month[sales_col], marker='o', color='blue', linewidth=2)
    peak_period = str(peak_year) + '-' + peak_month
    plt.axvline(peak_period, color='red', linestyle='--', label=f'Peak Sales: {peak_month} {peak_year} ({peak_sales})')

    plt.title("📈 Sales Trend Over Time with Peak Period Highlighted", fontsize=16)
    plt.xlabel("Year-Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return peak_year, peak_month, peak_sales
