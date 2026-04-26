import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from src.data.financial_data import load_financial_data

class FinancialAnomalyDetector:
    def __init__(self, csv_path: str, contamination: float = 0.02, max_history: int = 5000):
        self.csv_path = csv_path
        self.contamination = contamination
        self.max_history = max_history
        self.iso_forest = IsolationForest(contamination=contamination, random_state=42)
        
        self.le_area = LabelEncoder()
        self.le_type = LabelEncoder()
        
        # Diccionarios para almacenar las métricas de la regla de 3 sigmas
        self.stats_by_area = {}
        self.stats_by_type = {}
        self.stats_global = {}
        
        self._fit()

    def _fit(self):
        """Entrena el Isolation Forest y calcula métricas de 3-Sigma con datos históricos."""
        df = load_financial_data(self.csv_path)
        if df.empty:
            return

        # ---- 0. Límite de Historial ----
        if len(df) > self.max_history:
            df = df.tail(self.max_history).copy()

        # ---- 1. Preparación para Regla de 3 Sigmas ----
        self.stats_global = {
            'mean': df['Amount_num'].mean(),
            'std': df['Amount_num'].std()
        }
        
        for area, group in df.groupby('Area'):
            self.stats_by_area[area] = {
                'mean': group['Amount_num'].mean(),
                'std': group['Amount_num'].std(),
                'count': len(group)
            }
            
        for t, group in df.groupby('Type'):
            self.stats_by_type[t] = {
                'mean': group['Amount_num'].mean(),
                'std': group['Amount_num'].std(),
                'count': len(group)
            }

        # ---- 2. Preparación para Isolation Forest ----
        df_clean = df.dropna(subset=['Date_dt', 'Amount_num']).copy()
        if df_clean.empty:
            return

        df_clean['Area_encoded'] = self.le_area.fit_transform(df_clean['Area'])
        df_clean['Type_encoded'] = self.le_type.fit_transform(df_clean['Type'])
        
        df_clean['Month'] = df_clean['Date_dt'].dt.month
        df_clean['DayOfWeek'] = df_clean['Date_dt'].dt.dayofweek
        
        features = df_clean[['Amount_num', 'Area_encoded', 'Type_encoded', 'Month', 'DayOfWeek']]
        self.iso_forest.fit(features)

    def predict(self, date_str: str, amount_num: float, area: str, type_val: str) -> tuple[bool, list[str]]:
        """
        Analiza si una transacción es anómala (combinando 3-Sigma e Isolation Forest).
        Devuelve (es_anomalo, lista_de_razones).
        """
        reasons = []
        
        # ---- 1. Evaluación 3-Sigma ----
        g_mean = self.stats_global.get('mean', 0)
        g_std = self.stats_global.get('std', 0)
        if g_std > 0 and abs(amount_num - g_mean) > 3 * g_std:
            reasons.append(f"Regla 3-Sigma: Cantidad global atípica (media: {g_mean:.2f}€)")
            
        if area in self.stats_by_area:
            a_mean = self.stats_by_area[area]['mean']
            a_std = self.stats_by_area[area]['std']
            if self.stats_by_area[area]['count'] >= 3 and pd.notna(a_std) and a_std > 0:
                if abs(amount_num - a_mean) > 3 * a_std:
                    reasons.append(f"Regla 3-Sigma: Cantidad atípica para categoría '{area}' (media: {a_mean:.2f}€)")
        else:
            reasons.append(f"Regla 3-Sigma: Categoría '{area}' nunca antes vista.")
            
        if type_val in self.stats_by_type:
            t_mean = self.stats_by_type[type_val]['mean']
            t_std = self.stats_by_type[type_val]['std']
            if self.stats_by_type[type_val]['count'] >= 3 and pd.notna(t_std) and t_std > 0:
                if abs(amount_num - t_mean) > 3 * t_std:
                    reasons.append(f"Regla 3-Sigma: Cantidad atípica para tipo '{type_val}' (media: {t_mean:.2f}€)")

        # ---- 2. Evaluación Isolation Forest ----
        try:
            area_enc = self.le_area.transform([area])[0]
        except ValueError:
            area_enc = -1
            
        try:
            type_enc = self.le_type.transform([type_val])[0]
        except ValueError:
            type_enc = -1
            
        try:
            dt = pd.to_datetime(date_str, format='%d/%m/%Y')
            month = dt.month
            day_of_week = dt.dayofweek
        except ValueError:
            month = 1
            day_of_week = 0

        features = np.array([[amount_num, area_enc, type_enc, month, day_of_week]])
        iso_pred = self.iso_forest.predict(features)[0]
        
        if iso_pred == -1:
            reasons.append("Isolation Forest: Patrón multivariante/temporal inusual detectado.")
            
        is_anomalous = len(reasons) > 0
        return is_anomalous, reasons

    def reload(self):
        """Reentrena el modelo actualizando con los últimos datos."""
        self._fit()
