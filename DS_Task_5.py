# Traffic Accident Data Analysis - Complete Implementation
# Designed for Google Colab
# ============================================================================
# PART 1: SETUP AND INSTALLATIONS
# ============================================================================
# Install required packages
!pip install folium plotly kaleido opendatasets
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')
# Set style
plt.style.use('default')
sns.set_palette("husl")
print("🚀 Setup Complete! All libraries loaded successfully.")
# ============================================================================
# PART 2: DATA LOADING AND PREPARATION
# ============================================================================
# Method 1: Direct download (if available)
# Download dataset using opendatasets
import opendatasets as od
od.download("https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents")
# Load the dataset
try:
    df = pd.read_csv('/content/us-accidents/US_Accidents_March23.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Dataset shape: {df.shape}")
except:
    print("⚠️ Please upload the dataset manually or check the path")
    # Alternative: Upload manually to Colab
    # from google.colab import files
    # uploaded = files.upload()
    # df = pd.read_csv(list(uploaded.keys())[0])
# ============================================================================
# PART 3: EXPLORATORY DATA ANALYSIS
# ============================================================================
def basic_info():
    """Display basic dataset information"""
    print("=" * 60)
    print("📋 DATASET OVERVIEW")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\n🔍 Column Information:")
    print(df.info())
    print("\n📊 Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
    print("\n🎯 Target Variable Distribution (Severity):")
    print(df['Severity'].value_counts().sort_index())
# Data preprocessing
def preprocess_data():
    """Clean and prepare data for analysis"""
    global df
    print("🧹 Preprocessing data...")
    # Convert datetime columns
    datetime_cols = ['Start_Time', 'End_Time', 'Weather_Timestamp']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # Extract time features
    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.day_name()
    df['Month'] = df['Start_Time'].dt.month
    df['Year'] = df['Start_Time'].dt.year
    df['MonthName'] = df['Start_Time'].dt.month_name()
    # Clean categorical variables
    categorical_cols = ['Weather_Condition', 'Sunrise_Sunset', 'Civil_Twilight']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    # Create rush hour indicator
    df['Rush_Hour'] = df['Hour'].apply(lambda x: 'Yes' if x in [7,8,9,16,17,18] else 'No')
    # Create weekend indicator
    df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 'Yes' if x in ['Saturday', 'Sunday'] else 'No')
    print("✅ Data preprocessing completed!")
    return df
# Run basic analysis
basic_info()
df = preprocess_data()
# ============================================================================
# PART 4: TIME PATTERN ANALYSIS
# ============================================================================
def analyze_time_patterns():
    """Analyze accident patterns by time"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accidents by Hour of Day', 'Accidents by Day of Week',
                       'Accidents by Month', 'Rush Hour vs Normal Hours'),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "domain"}]]  # domain allows pie charts
    )
    # Hour analysis
    hour_data = df.groupby('Hour').size().reset_index(name='Count')
    fig.add_trace(
        go.Scatter(x=hour_data['Hour'], y=hour_data['Count'],
                  mode='lines+markers', name='Hourly Accidents',
                  line=dict(color='#FF6B6B', width=3)),
        row=1, col=1
    )
    # Day of week analysis
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_data = df.groupby('DayOfWeek').size().reindex(day_order).reset_index(name='Count')
    fig.add_trace(
        go.Bar(x=day_data['DayOfWeek'], y=day_data['Count'],
               name='Daily Accidents', marker_color='#4ECDC4'),
        row=1, col=2
    )
    # Month analysis
    month_data = df.groupby('MonthName').size().reset_index(name='Count')
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_data = month_data.set_index('MonthName').reindex(month_order).reset_index()
    fig.add_trace(
        go.Bar(x=month_data['MonthName'], y=month_data['Count'],
               name='Monthly Accidents', marker_color='#45B7D1'),
        row=2, col=1
    )
    # Rush hour analysis
    rush_data = df.groupby('Rush_Hour').size().reset_index(name='Count')
    fig.add_trace(
        go.Pie(labels=rush_data['Rush_Hour'], values=rush_data['Count'],
               name='Rush Hour Distribution'),
        row=2, col=2
    )
    fig.update_layout(height=800, title_text="🕐 Time Pattern Analysis", showlegend=False)
    fig.show()
    # Statistical insights
    print("📊 TIME PATTERN INSIGHTS:")
    print("-" * 50)
    peak_hour = df.groupby('Hour').size().idxmax()
    peak_day = df.groupby('DayOfWeek').size().idxmax()
    peak_month = df.groupby('Month').size().idxmax()
    print(f"🔥 Peak accident hour: {peak_hour}:00")
    print(f"🔥 Peak accident day: {peak_day}")
    print(f"🔥 Peak accident month: {peak_month}")
    print(f"🚗 Rush hour accidents: {(df['Rush_Hour'] == 'Yes').sum():,}")
    print(f"🏠 Weekend accidents: {(df['Is_Weekend'] == 'Yes').sum():,}")
analyze_time_patterns()
# ============================================================================
# PART 5: WEATHER AND ROAD CONDITIONS ANALYSIS
# ============================================================================
def analyze_weather_conditions():
    """Analyze impact of weather on accidents"""
    # Weather condition analysis
    weather_counts = df['Weather_Condition'].value_counts().head(10)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top 10 Weather Conditions', 'Severity by Weather',
                       'Visibility Impact', 'Temperature Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    # Weather conditions
    fig.add_trace(
        go.Bar(x=weather_counts.values, y=weather_counts.index,
               orientation='h', name='Weather Frequency',
               marker_color='skyblue'),
        row=1, col=1
    )
    # Severity by weather (top 5 weather conditions)
    top_weather = weather_counts.head(5).index
    severity_weather = df[df['Weather_Condition'].isin(top_weather)].groupby(['Weather_Condition', 'Severity']).size().unstack(fill_value=0)
    for severity in severity_weather.columns:
        fig.add_trace(
            go.Bar(x=severity_weather.index, y=severity_weather[severity],
                   name=f'Severity {severity}'),
            row=1, col=2
        )
    # Visibility analysis
    if 'Visibility(mi)' in df.columns:
        visibility_data = df.groupby(pd.cut(df['Visibility(mi)'], bins=5))['Severity'].mean()
        fig.add_trace(
            go.Scatter(x=[str(x) for x in visibility_data.index], y=visibility_data.values,
                      mode='lines+markers', name='Avg Severity by Visibility'),
            row=2, col=1
        )
    # Temperature analysis
    if 'Temperature(F)' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['Temperature(F)'].dropna(), name='Temperature Distribution',
                        marker_color='orange', opacity=0.7),
            row=2, col=2
        )
    fig.update_layout(height=800, title_text="🌤️ Weather Conditions Analysis")
    fig.show()
    # Weather insights
    print("🌦️ WEATHER INSIGHTS:")
    print("-" * 40)
    print(f"Most common weather: {weather_counts.index[0]} ({weather_counts.iloc[0]:,} accidents)")
    if 'Visibility(mi)' in df.columns:
        avg_visibility = df['Visibility(mi)'].mean()
        print(f"Average visibility: {avg_visibility:.2f} miles")
    if 'Temperature(F)' in df.columns:
        avg_temp = df['Temperature(F)'].mean()
        print(f"Average temperature: {avg_temp:.1f}°F")
def analyze_road_conditions():
    """Analyze road infrastructure impact"""
    # Road feature analysis
    road_features = ['Traffic_Signal', 'Stop', 'Railway', 'Roundabout', 'Station', 'Traffic_Calming']
    road_data = []
    for feature in road_features:
        if feature in df.columns:
            feature_impact = df.groupby(feature)['Severity'].agg(['count', 'mean']).reset_index()
            feature_impact['Feature'] = feature
            road_data.append(feature_impact)
    if road_data:
        road_df = pd.concat(road_data, ignore_index=True)
        fig = px.scatter(road_df, x='count', y='mean', color='Feature', size='count',
                        title='🛣️ Road Features Impact on Accident Severity',
                        labels={'count': 'Number of Accidents', 'mean': 'Average Severity'})
        fig.show()
    # Sunrise/Sunset analysis
    if 'Sunrise_Sunset' in df.columns:
        daylight_analysis = df.groupby('Sunrise_Sunset').agg({
            'Severity': ['count', 'mean']
        }).round(2)
        print("☀️ DAYLIGHT CONDITIONS:")
        print("-" * 30)
        print(daylight_analysis)
analyze_weather_conditions()
analyze_road_conditions()
# ============================================================================
# PART 6: GEOSPATIAL ANALYSIS - ACCIDENT HOTSPOTS
# ============================================================================
def create_accident_hotspots():
    """Create interactive maps showing accident hotspots"""
    # State-level analysis
    if 'State' in df.columns:
        state_data = df.groupby('State').agg({
            'ID': 'count',
            'Severity': 'mean'
        }).round(2).reset_index()
        state_data.columns = ['State', 'Accident_Count', 'Avg_Severity']
        # Choropleth map
        fig = px.choropleth(
            state_data,
            locations='State',
            color='Accident_Count',
            locationmode='USA-states',
            title='🗺️ Accident Distribution by State',
            color_continuous_scale='Reds',
            scope='usa'
        )
        fig.show()
        # Top 10 states
        top_states = state_data.nlargest(10, 'Accident_Count')
        fig = px.bar(top_states, x='State', y='Accident_Count',
                    title='🔥 Top 10 States by Accident Count',
                    color='Avg_Severity', color_continuous_scale='viridis')
        fig.show()
    # City-level analysis
    if 'City' in df.columns:
        city_data = df.groupby('City').agg({
            'ID': 'count',
            'Severity': 'mean'
        }).round(2).reset_index()
        city_data.columns = ['City', 'Accident_Count', 'Avg_Severity']
        top_cities = city_data.nlargest(15, 'Accident_Count')
        fig = px.bar(top_cities, x='Accident_Count', y='City',
                    title='🏙️ Top 15 Cities by Accident Count',
                    orientation='h', color='Avg_Severity',
                    color_continuous_scale='plasma')
        fig.show()
    # Create heatmap (sample of data for performance)
    if 'Start_Lat' in df.columns and 'Start_Lng' in df.columns:
        sample_df = df.sample(n=min(10000, len(df)))  # Sample for performance
        # Create folium map
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        # Add heatmap
        heat_data = [[row['Start_Lat'], row['Start_Lng']] for idx, row in sample_df.iterrows()
                     if pd.notna(row['Start_Lat']) and pd.notna(row['Start_Lng'])]
        HeatMap(heat_data, radius=10, blur=15, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(m)
        # Save map
        m.save('/content/accident_heatmap.html')
        print("🗺️ Interactive heatmap saved as 'accident_heatmap.html'")
        # Display map in Colab
        from IPython.display import IFrame
        IFrame('/content/accident_heatmap.html', width=700, height=500)
create_accident_hotspots()
# ============================================================================
# PART 7: CORRELATION AND PATTERN ANALYSIS
# ============================================================================
def advanced_pattern_analysis():
    """Perform advanced statistical analysis"""
    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('🔥 Correlation Matrix - Accident Features')
    plt.tight_layout()
    plt.show()
    # Severity analysis by multiple factors
    if all(col in df.columns for col in ['Hour', 'Is_Weekend', 'Weather_Condition']):
        severity_analysis = df.groupby(['Hour', 'Is_Weekend'])['Severity'].agg(['count', 'mean']).reset_index()
        fig = px.scatter(severity_analysis, x='Hour', y='mean', size='count',
                        color='Is_Weekend', title='📊 Accident Severity Patterns',
                        labels={'mean': 'Average Severity', 'count': 'Number of Accidents'})
        fig.show()
    # Statistical summary by severity
    severity_summary = df.groupby('Severity').agg({
        'Temperature(F)': 'mean',
        'Humidity(%)': 'mean',
        'Visibility(mi)': 'mean',
        'Wind_Speed(mph)': 'mean'
    }).round(2)
    print("📈 SEVERITY ANALYSIS:")
    print("-" * 40)
    print(severity_summary)
advanced_pattern_analysis()
# ============================================================================
# PART 8: BUSINESS INSIGHTS AND RECOMMENDATIONS
# ============================================================================
def generate_insights():
    """Generate key insights and recommendations"""
    print("=" * 80)
    print("🎯 KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 80)
    # Time-based insights
    peak_hour = df.groupby('Hour').size().idxmax()
    peak_accidents = df.groupby('Hour').size().max()
    print("⏰ TIME PATTERNS:")
    print(f"• Peak accident time: {peak_hour}:00 with {peak_accidents:,} accidents")
    print(f"• Rush hour contributes to {(df['Rush_Hour'] == 'Yes').sum() / len(df) * 100:.1f}% of accidents")
    print(f"• Weekend accidents: {(df['Is_Weekend'] == 'Yes').sum() / len(df) * 100:.1f}% of total")
    # Weather insights
    if 'Weather_Condition' in df.columns:
        top_weather = df['Weather_Condition'].value_counts().head(1)
        print("\n🌤️ WEATHER IMPACT:")
        print(f"• Most accidents occur in: {top_weather.index[0]} ({top_weather.iloc[0]:,} cases)")
        if 'Visibility(mi)' in df.columns:
            low_visibility = (df['Visibility(mi)'] < 2).sum()
            print(f"• Low visibility accidents (<2 miles): {low_visibility:,}")
    # Geographic insights
    if 'State' in df.columns:
        top_state = df['State'].value_counts().head(1)
        print("\n🗺️ GEOGRAPHIC HOTSPOTS:")
        print(f"• Highest accident state: {top_state.index[0]} ({top_state.iloc[0]:,} accidents)")
    if 'City' in df.columns:
        top_city = df['City'].value_counts().head(1)
        print(f"• Highest accident city: {top_city.index[0]} ({top_city.iloc[0]:,} accidents)")
    # Severity insights
    severity_dist = df['Severity'].value_counts().sort_index()
    print(f"\n⚠️ SEVERITY DISTRIBUTION:")
    for severity, count in severity_dist.items():
        percentage = count / len(df) * 100
        print(f"• Severity {severity}: {count:,} accidents ({percentage:.1f}%)")
    print("\n" + "=" * 80)
    print("💡 STRATEGIC RECOMMENDATIONS")
    print("=" * 80)
    recommendations = [
        "🚦 Increase traffic enforcement during peak hours (7-9 AM, 4-6 PM)",
        "🌧️ Implement weather-based traffic management systems",
        "📍 Focus safety improvements on identified hotspot cities/states",
        "🚨 Deploy more emergency services during high-risk time periods",
        "📱 Develop real-time traffic safety alerts for drivers",
        "🛣️ Improve road infrastructure at accident-prone intersections",
        "📊 Regular monitoring and analysis for pattern changes",
        "🎓 Public awareness campaigns targeting high-risk scenarios"
    ]
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    print(f"\n📈 ANALYSIS COMPLETED - {len(df):,} records processed successfully!")
generate_insights()
# ============================================================================
# PART 9: EXPORT RESULTS
# ============================================================================
def export_results():
    """Export analysis results"""
    print("\n📄 EXPORTING RESULTS...")
    # Summary statistics
    summary_stats = {
        'Total_Accidents': len(df),
        'Date_Range': f"{df['Start_Time'].min().date()} to {df['Start_Time'].max().date()}",
        'States_Covered': df['State'].nunique() if 'State' in df.columns else 'N/A',
        'Cities_Covered': df['City'].nunique() if 'City' in df.columns else 'N/A',
        'Peak_Hour': df.groupby('Hour').size().idxmax(),
        'Most_Common_Weather': df['Weather_Condition'].mode()[0] if 'Weather_Condition' in df.columns else 'N/A'
    }
    # Save summary
    summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
    summary_df.to_csv('/content/accident_analysis_summary.csv', index=False)
    # Top insights
    if 'State' in df.columns:
        state_summary = df.groupby('State').agg({
            'ID': 'count',
            'Severity': 'mean'
        }).round(2).reset_index()
        state_summary.columns = ['State', 'Total_Accidents', 'Avg_Severity']
        state_summary.to_csv('/content/state_analysis.csv', index=False)
    if 'City' in df.columns:
        city_summary = df.groupby('City').agg({
            'ID': 'count',
            'Severity': 'mean'
        }).round(2).reset_index().head(50)
        city_summary.columns = ['City', 'Total_Accidents', 'Avg_Severity']
        city_summary.to_csv('/content/top_cities_analysis.csv', index=False)
    print("✅ Results exported:")
    print("   • accident_analysis_summary.csv")
    print("   • state_analysis.csv")
    print("   • top_cities_analysis.csv")
    print("   • accident_heatmap.html")
export_results()
print("\n" + "=" * 80)
print("🎉 ANALYSIS COMPLETE!")
print("=" * 80)
print("✅ All visualizations generated")
print("✅ Insights extracted")
print("✅ Files exported")
print("✅ Ready for presentation!")
print("\nYou now have a comprehensive traffic accident analysis that would impress any employer! 🚀")
