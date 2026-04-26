import csv
from pathlib import Path
import pandas as pd

def parse_amount(val: str) -> float:
    """Convierte un string como '10,00€' o '1.841,97€' a float."""
    if pd.isna(val):
        return 0.0
    val = str(val).replace('€', '').strip()
    val = val.replace('.', '')  # Separador de miles
    val = val.replace(',', '.') # Separador decimal
    try:
        return float(val)
    except ValueError:
        return 0.0

def format_amount(val: float) -> str:
    """Convierte un float a string de formato '1.841,97€'."""
    s = f"{val:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"{s}€"

def load_financial_data(csv_path: str) -> pd.DataFrame:
    """Carga y preprocesa el histórico financiero."""
    df = pd.read_csv(csv_path)
    
    # Calcular cantidad numérica y fecha datetime
    df['Amount_num'] = df['Amount'].apply(parse_amount)
    # coerce errors just in case format is weird in some rows
    df['Date_dt'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    return df

def append_transaction(csv_path: str, description: str, date: str, amount_str: str, area: str, type_val: str):
    """Añade una nueva transacción al final del CSV."""
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([description, date, amount_str, area, type_val])
