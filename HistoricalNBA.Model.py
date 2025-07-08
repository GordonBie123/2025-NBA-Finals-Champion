# Import all Libraries and Modules
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

# Loading Data and Past Championship Outcomes

# Championship Results:
CHAMPIONSHIP_RESULTS = {
    2020: 'LAL',  # Los Angeles Lakers
    2021: 'MIL',  # Milwaukee Bucks
    2022: 'GSW',  # Golden State Warriors
    2023: 'DEN',  # Denver Nuggets
    2024: 'BOS'   # Boston Celtics
}

def load_historical_data():
    """Load and combine all historical playoff data"""
    import os
    
    # Set working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check what files are actually available
    available_files = [f for f in os.listdir('.') if f.endswith('nbaplayoff.csv')]
    
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    all_data = []
    
    for year in years:
        filename = f'{year}nbaplayoff.csv'
        
        if filename in available_files:
            try:
                df = pd.read_csv(filename)
                df['Year'] = year
                
                # Add championship outcome for historical years
                if year in CHAMPIONSHIP_RESULTS:
                    df['Champion'] = (df['Team'] == CHAMPIONSHIP_RESULTS[year]).astype(int)
                else:
                    df['Champion'] = np.nan  # 2025 - to be predicted
                
                all_data.append(df)
                
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        else:
            print(f"❌ File not found: {filename}")
    
    if len(all_data) == 0:
        print("❌ No data files found! Please check file names and location.")
        raise FileNotFoundError("No playoff data files found")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# Engineer Features

def create_advanced_features(df):
    """Create sophisticated basketball analytics features"""
    
    # Individual Player Features
    
    # Handle different column sets (2020-2024 vs 2025)
    if 'PER' in df.columns:  # 2025 advanced stats
        df['Efficiency_Score'] = (df['PER'] * df['TS%'] * df['WS/48']).fillna(0)
        df['Impact_Score'] = (df['BPM'] * df['VORP']).fillna(0)
        df['Usage_Efficiency'] = (df['TS%'] / df['USG%']).fillna(0)
        
    else:  # 2020-2024 basic stats - create advanced metrics
        # Calculate advanced metrics from basic stats
        df['PPG'] = df['PTS'] / df['G']
        df['RPG'] = df['TRB'] / df['G'] 
        df['APG'] = df['AST'] / df['G']
        df['SPG'] = df['STL'] / df['G']
        df['BPG'] = df['BLK'] / df['G']
        df['TOPG'] = df['TOV'] / df['G']
        df['MPG'] = df['MP'] / df['G']
        
        # Efficiency Metrics
        df['True_Shooting'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
        df['Effective_FG'] = (df['FG'] + 0.5 * df['3P']) / df['FGA']
        df['Usage_Rate'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MP']
        df['Assist_Rate'] = df['AST'] / (df['FGA'] + 0.44 * df['FTA'] + df['AST'] + df['TOV'])x
        df['Assist_TO_Ratio'] = df['AST'] / df['TOV']  # Key championship metric
        
        # Versatility metrics
        df['Statistical_Versatility'] = (
            (df['PPG'] > 10).astype(int) + 
            (df['RPG'] > 5).astype(int) + 
            (df['APG'] > 3).astype(int) + 
            (df['SPG'] > 1).astype(int) + 
            (df['BPG'] > 0.5).astype(int)
        )
    
    # Experience and Age Features
    
    # Age-based features
    df['Age_Group'] = pd.cut(df['Age'], 
                           bins=[0, 23, 27, 30, 35, 50], 
                           labels=['Young', 'Developing', 'Prime', 'Veteran', 'Elder'],
                           include_lowest=True)
    
    # Experience proxy (games played)
    df['Experience_Level'] = pd.cut(df['G'], 
                                  bins=[0, 5, 10, 15, 20, 30], 
                                  labels=['Limited', 'Rotation', 'Regular', 'Key', 'Star'],
                                  include_lowest=True)
    
    # Playoff veteran indicator (experienced players in pressure situations)
    df['Playoff_Veteran'] = (df['Age'] >= 28).astype(int)
    
    # Position-Based Features
    
    # Simplified position categories
    df['Position_Category'] = df['Pos'].apply(lambda x: categorize_position(x) if pd.notna(x) else 'Unknown')
    
    # Performance Context Features
    
    # Minutes played importance
    df['Minutes_Importance'] = df['MP'] / df['MP'].max() if 'MP' in df.columns else 0
    
    # Scoring responsibility
    if 'PTS' in df.columns:
        df['Team_Scoring_Share'] = df.groupby(['Team', 'Year'])['PTS'].transform(lambda x: x / x.sum())
    
    return df

def categorize_position(pos):
    """Categorize positions into simplified groups"""
    if pd.isna(pos):
        return 'Unknown'
    pos = str(pos).upper()
    if 'PG' in pos or 'POINT' in pos:
        return 'Guard'
    elif 'SG' in pos or 'SF' in pos or 'SHOOTING' in pos or 'SMALL' in pos:
        return 'Wing'  
    elif 'PF' in pos or 'C' in pos or 'POWER' in pos or 'CENTER' in pos:
        return 'Big'
    else:
        return 'Combo'

# SECTION 3: Team Chemistry Features

def create_team_features(df):
    """Create team-level aggregated features with championship-specific patterns"""
    
    team_features = []
    
    for (team, year), group in df.groupby(['Team', 'Year']):
        if len(group) == 0:
            continue
            
        team_stats = {'Team': team, 'Year': year}
        
        # Basic aggregations
        numeric_cols = group.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Rk', 'Champion', 'Year']:
                team_stats[f'{col}_mean'] = group[col].mean()
                team_stats[f'{col}_std'] = group[col].std()
                team_stats[f'{col}_max'] = group[col].max()
                team_stats[f'{col}_sum'] = group[col].sum()
        
        # Create PPG if it doesn't exist (handle both 2020-2024 and 2025 data)
        if 'PPG' not in group.columns and 'PTS' in group.columns and 'G' in group.columns:
            group = group.copy()
            group['PPG'] = group['PTS'] / group['G']
        
        # Star-Role Player Balance (Critical for championships)
        if 'PPG' in group.columns:
            star_players = group[group['PPG'] >= 18]  # Stars (18+ PPG)
            secondary_stars = group[(group['PPG'] >= 12) & (group['PPG'] < 18)]  # Secondary scorers
            role_players = group[(group['PPG'] >= 6) & (group['PPG'] < 12)]  # Role players
            
            team_stats['Star_Count'] = len(star_players)
            team_stats['Secondary_Star_Count'] = len(secondary_stars)
            team_stats['Role_Player_Count'] = len(role_players)
            team_stats['Star_Quality'] = star_players['PPG'].max() if len(star_players) > 0 else 0
            team_stats['Scoring_Balance'] = 1 / (group['PPG'].std() + 1)  # Lower std = better balance
            
            # Championship DNA: Need superstar + depth
            team_stats['Has_Superstar'] = int(group['PPG'].max() >= 22)
            team_stats['Has_Second_Option'] = int(group['PPG'].nlargest(2).iloc[1] >= 15 if len(group) >= 2 else False)
            team_stats['Depth_Quality'] = (group['PPG'] >= 8).sum()  # Players contributing meaningfully
        else:
            # Fallback values if PPG can't be calculated
            team_stats['Star_Count'] = 0
            team_stats['Secondary_Star_Count'] = 0
            team_stats['Role_Player_Count'] = 0
            team_stats['Star_Quality'] = 0
            team_stats['Scoring_Balance'] = 0
            team_stats['Has_Superstar'] = 0
            team_stats['Has_Second_Option'] = 0
            team_stats['Depth_Quality'] = 0
        
        # Age and Experience Balance (Championship teams need mix)
        team_stats['Average_Age'] = group['Age'].mean()
        team_stats['Age_Variance'] = group['Age'].std()
        team_stats['Veteran_Count'] = (group['Age'] >= 30).sum()
        team_stats['Young_Player_Count'] = (group['Age'] <= 23).sum()
        team_stats['Prime_Player_Count'] = ((group['Age'] >= 25) & (group['Age'] <= 30)).sum()
        team_stats['Playoff_Veteran_Count'] = group['Playoff_Veteran'].sum()
        team_stats['Experience_Balance'] = team_stats['Playoff_Veteran_Count'] / len(group)
        
        # Team depth features
        if 'MP' in group.columns:
            team_stats['Depth_Score'] = (group['MP'] > 15).sum()  # Players with meaningful minutes
            team_stats['Star_Power'] = group['MP'].nlargest(3).mean()  # Top 3 players minutes
            team_stats['Bench_Strength'] = group['MP'].nsmallest(len(group)//2).mean()
            team_stats['Minutes_Distribution'] = 1 / (group['MP'].std() + 1)  # Balanced minutes
        
        # Position balance (Championship teams need versatility)
        if 'Position_Category' in group.columns:
            pos_counts = group['Position_Category'].value_counts()
            for pos in ['Guard', 'Wing', 'Big', 'Combo']:
                team_stats[f'{pos}_Count'] = pos_counts.get(pos, 0)
            
            # Position balance score
            guard_count = pos_counts.get('Guard', 0)
            wing_count = pos_counts.get('Wing', 0)
            big_count = pos_counts.get('Big', 0)
            team_stats['Position_Balance'] = min(guard_count, wing_count, big_count)

        # Efficiency Metrics
        if 'True_Shooting' in group.columns:
            team_stats['Team_True_Shooting'] = group['True_Shooting'].mean()
            team_stats['Team_Assist_TO_Ratio'] = group['Assist_TO_Ratio'].mean()
            team_stats['Efficiency_Consistency'] = 1 / (group['True_Shooting'].std() + 0.1)
        
        # Championship-Level Benchmarks
        if 'FG%' in group.columns:
            team_stats['Elite_Shooting'] = int(group['FG%'].mean() >= 0.46)
        else:
            team_stats['Elite_Shooting'] = 0
            
        if 'STL' in group.columns and 'BLK' in group.columns:
            team_stats['Elite_Defense'] = int((group['STL'].sum() + group['BLK'].sum()) >= 20)
        else:
            team_stats['Elite_Defense'] = 0
            
        # Use PPG if available, otherwise fallback
        if 'PPG' in group.columns:
            team_stats['Elite_Depth'] = int((group['PPG'] >= 8).sum() >= 7)
        elif 'PTS' in group.columns and 'G' in group.columns:
            ppg_calc = group['PTS'] / group['G']
            team_stats['Elite_Depth'] = int((ppg_calc >= 8).sum() >= 7)
        else:
            team_stats['Elite_Depth'] = 0
        
        # Two-Way Impact (players who contribute on both ends)
        if 'PPG' in group.columns and 'STL' in group.columns and 'BLK' in group.columns:
            two_way_players = group[(group['PPG'] >= 10) & ((group['STL'] + group['BLK']) >= 0.8)]
            team_stats['Two_Way_Player_Count'] = len(two_way_players)
        elif 'PTS' in group.columns and 'G' in group.columns and 'STL' in group.columns and 'BLK' in group.columns:
            # Calculate PPG on the fly if needed
            group_copy = group.copy()
            group_copy['PPG_temp'] = group_copy['PTS'] / group_copy['G']
            two_way_players = group_copy[(group_copy['PPG_temp'] >= 10) & ((group_copy['STL'] + group_copy['BLK']) >= 0.8)]
            team_stats['Two_Way_Player_Count'] = len(two_way_players)
        else:
            team_stats['Two_Way_Player_Count'] = 0
        
        # Team composition features
        team_stats['Roster_Size'] = len(group)
        
        # Championship DNA Score (combination of key factors)
        championship_factors = [
            team_stats.get('Has_Superstar', 0),
            team_stats.get('Has_Second_Option', 0),
            team_stats.get('Elite_Shooting', 0),
            team_stats.get('Elite_Defense', 0),
            team_stats.get('Elite_Depth', 0),
            int(team_stats.get('Experience_Balance', 0) >= 0.3),  # At least 30% veterans
            int(team_stats.get('Position_Balance', 0) >= 2)  # Balanced positions
        ]
        team_stats['Championship_DNA_Score'] = sum(championship_factors)
        
        # Championship target
        team_stats['Champion'] = group['Champion'].iloc[0] if 'Champion' in group.columns else np.nan
        
        team_features.append(team_stats)
    
    return pd.DataFrame(team_features)

# Model Training

def prepare_training_data(team_df):
    """Prepare data for machine learning with feature selection"""
    
    # Remove 2025 data for training (prediction target)
    train_data = team_df[team_df['Year'] != 2025].copy()
    predict_data = team_df[team_df['Year'] == 2025].copy()
    
    # Remove non-feature columns
    feature_cols = [col for col in train_data.columns if col not in 
                   ['Team', 'Year', 'Champion']]
    
    X_train = train_data[feature_cols]
    y_train = train_data['Champion']
    X_predict = predict_data[feature_cols]
    
    # Handle missing values
    X_train = X_train.fillna(0)
    X_predict = X_predict.fillna(0)
    
    # Handle infinite values
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_predict = X_predict.replace([np.inf, -np.inf], 0)
    
    # Feature Selection
    try:
        selector = SelectKBest(score_func=f_classif, k=min(50, X_train.shape[1]//2))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_predict_selected = selector.transform(X_predict)
        
        # Get selected feature names
        selected_features = X_train.columns[selector.get_support()]
        
        return pd.DataFrame(X_train_selected, columns=selected_features), y_train, pd.DataFrame(X_predict_selected, columns=selected_features), predict_data
    
    except Exception as e:
        return X_train, y_train, X_predict, predict_data

def train_ensemble_model(X_train, y_train):
    """QUICK WIN #3: Train ensemble of multiple models"""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Ensemble of 5 different algorithms for feature selection
    models = {
        'Random_Forest': RandomForestClassifier(
            n_estimators=300,  # More trees
            random_state=42, 
            max_depth=12,  # Deeper trees
            class_weight='balanced',
            min_samples_split=5,
            min_samples_leaf=2
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=8,
            learning_rate=0.05  # Slower learning for better generalization
        ),
        'Logistic_Regression': LogisticRegression(
            random_state=42, 
            max_iter=2000,
            class_weight='balanced',
            C=0.1  # Regularization
        ),
        'SVM': SVC(
            random_state=42,
            class_weight='balanced',
            probability=True,  # Enable probability predictions
            kernel='rbf',
            C=1.0
        ),
        'Neural_Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            random_state=42,
            max_iter=1000,
            alpha=0.01  # Regularization
        )
    }
    
    # Train individual models and collect their predictions
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        try:
            # Fit the model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate performance
            train_score = model.score(X_train_scaled, y_train)
            model_scores[name] = train_score
            trained_models[name] = model
            
        except Exception as e:
            continue
    
    # Create Voting Ensemble (combines all models)
    if len(trained_models) >= 3:
        # Create ensemble
        ensemble_models = [(name, model) for name, model in trained_models.items()]
        voting_ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'  # Use probability averages
        )
        
        # Train ensemble
        voting_ensemble.fit(X_train_scaled, y_train)
        ensemble_score = voting_ensemble.score(X_train_scaled, y_train)
        
        # Use ensemble as final model
        best_model = voting_ensemble
        best_score = ensemble_score
        
    else:
        # Fallback to best individual model
        best_name = max(model_scores.items(), key=lambda x: x[1])[0] if model_scores else 'Random_Forest'
        best_model = trained_models.get(best_name)
        best_score = model_scores.get(best_name, 0)
    
    # Final fallback
    if best_model is None:
        best_model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            class_weight='balanced'
        )
        best_model.fit(X_train_scaled, y_train)
        best_score = best_model.score(X_train_scaled, y_train)
    
    return best_model, scaler, model_scores

# Main Execution

def main():
    print("🏀 ENHANCED NBA FINALS PREDICTION MODEL")
    print("=" * 50)
    
    # Load data
    df = load_historical_data()
    
    # Feature engineering
    df = create_advanced_features(df)
    
    # Team aggregation with championship-specific features
    team_df = create_team_features(df)
    
    # Prepare training data with feature selection
    X_train, y_train, X_predict, predict_teams = prepare_training_data(team_df)
    
    # Train ensemble models
    model, scaler, scores = train_ensemble_model(X_train, y_train)
    
    # Make predictions
    X_predict_scaled = scaler.transform(X_predict)
    predictions = model.predict_proba(X_predict_scaled)[:, 1]
    
    # Results
    results = predict_teams[['Team', 'Year']].copy()
    results['Championship_Probability'] = predictions
    results = results.sort_values('Championship_Probability', ascending=False)
    
    print("\n" + "=" * 50)
    print("🏆 2025 NBA FINALS PREDICTION RESULTS")
    print("=" * 50)
    
    # Filter for Finals teams
    finals_teams = results[results['Team'].isin(['OKC', 'IND'])].copy()
    
    if len(finals_teams) == 2:
        # NORMALIZE probabilities for head-to-head matchup
        okc_raw_prob = finals_teams[finals_teams['Team'] == 'OKC']['Championship_Probability'].iloc[0]
        ind_raw_prob = finals_teams[finals_teams['Team'] == 'IND']['Championship_Probability'].iloc[0]
        
        # Normalize so they add up to 100%
        total_prob = okc_raw_prob + ind_raw_prob
        okc_normalized = okc_raw_prob / total_prob
        ind_normalized = ind_raw_prob / total_prob
        
        print(f"\n🏆 HEAD-TO-HEAD FINALS PROBABILITIES:")
        print(f"OKC Thunder:     {okc_normalized:.1%} {'🏆 PREDICTED WINNER' if okc_normalized > 0.5 else '  Underdog'}")
        print(f"IND Pacers:      {ind_normalized:.1%} {'🏆 PREDICTED WINNER' if ind_normalized > 0.5 else '  Underdog'}")
        
        winner_team = 'OKC' if okc_normalized > ind_normalized else 'IND'
        winner_prob = max(okc_normalized, ind_normalized)
        
        print(f"\n🎯 FINAL PREDICTION: {winner_team} wins 2025 NBA Finals")
        print(f"📊 Head-to-Head Confidence: {winner_prob:.1%}")
        
        # Show confidence level
        if winner_prob > 0.70:
            confidence_level = "HIGH 🔥"
        elif winner_prob > 0.60:
            confidence_level = "MODERATE 📈"
        else:
            confidence_level = "LOW ⚠️"
        
        print(f"🎚️  Model Confidence Level: {confidence_level}")
        
    else:
        print("⚠️  Finals teams (OKC/IND) not found in prediction data")
    
    # Save model and results
    joblib.dump(model, 'enhanced_nba_model.pkl')
    joblib.dump(scaler, 'enhanced_scaler.pkl')
    results.to_csv('2025_enhanced_predictions.csv', index=False)
    
    print(f"\n💾 Model and results saved successfully")
    print("=" * 50)
    
    return model, scaler, results

if __name__ == "__main__":
    model, scaler, results = main()