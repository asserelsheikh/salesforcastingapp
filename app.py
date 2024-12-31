import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fpdf import FPDF
import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Sales Performance & Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
        body {
            background-color:rgb(195, 210, 248);
            font-family: serif;
        }

        .stApp {
            background-image: url('https://lpsonline.sas.upenn.edu/sites/default/files/2022-10/plpso-feratures-data-business.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-color: rgba(255, 255, 255, 0.5); 
            background-blend-mode: lighten;
        }

        h1, h2, h3 {
            color:rgb(15, 84, 131);
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }

        .stMarkdown {
            background-color: #fff;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            color: #181717;
        }

        .stSidebar {
            background-color: rgba(40, 41, 39, 0.46);
            color: #181717;
            padding: 1rem;
        }

        .stButton > button {
            background-color: rgb(39, 67, 68);
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background-color: rgb(34, 75, 99);
            transform: translateY(-2px);
        }

        .metric-card {
            background: rgba(196, 222, 244, 0.93);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
            color: #181717;
        }

        .metric-card:hover {
            transform: translateY(-5px);
        }

        .metric-card h4 {
            margin-bottom: 0.5rem;
            color: #333;
            font-size: 1.1rem;
        }

        .chart-container {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
            color: #181717;
        }

        .indicator {
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
            font-weight: bold;
            text-align: center;
            color: #181717;
        }

        .indicator-up {
            background-color: rgb(1, 6, 1); /* Darker green background */
            color: #1b5e20; /* Darker green text */
            border: 1px solid #1b5e20; /* Darker green border */
        }

        .indicator-down {
            background-color: rgb(5, 3, 4); /* Darker red background */
            color: #8e0000; /* Darker red text */
            border: 1px solid #8e0000; /* Darker red border */
        }

        .segment-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            color: #181717;
        }

        /* Sidebar customization */
        .sidebar .sidebar-content {
            background-color: #edeef5;  /* Light secondary background color */
        }
        
        /* Text color for all interactive components */
        .stText, .stButton, .stRadio, .stCheckbox, .stSelectbox {
            color: #181717;
        }

        /* Set link color to primary */
        a {
            color:rgb(19, 81, 122);
        }
        
        /* Make button background match primary theme */
        .stButton > button {
            background-color:rgb(17, 82, 125);
            color: white;
        }

        .stButton > button:hover {
            background-color:rgb(17, 82, 125);
        }

    </style>
""", unsafe_allow_html=True)

def validate_data(df):
    required_columns = ['Date', 'Hour', 'Total', 'Paid']
    optional_columns = ['CustomerID']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"

    for col in required_columns:
        try:
            if col == 'Date':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            return False, f"Error processing column '{col}': {str(e)}"

    if df[required_columns].isnull().any().any():
        return False, "Some required columns contain invalid or missing values."

    return True, "Data validation successful."


def create_metric_cards(metrics):
    cols = st.columns(len(metrics))
    for col, (title, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>{title}</h4>
                    <h1>{value}</h1>
                </div>
            """, unsafe_allow_html=True)

def create_visualizations(data):
    try:
        fig_daily = px.line(
            data.groupby('Date')['Total'].sum().reset_index(),
            x='Date',
            y='Total',
            title='Daily Sales Trend'
        )
        fig_daily.update_layout(title_text="Daily Sales Trend", title_x=0.5)

        fig_hourly = px.bar(
            data.groupby('Hour')['Total'].mean().reset_index(),
            x='Hour',
            y='Total',
            title='Average Sales by Hour'
        )
        fig_hourly.update_layout(title_text="Average Sales by Hour", title_x=0.5)

        # Render visualizations using Streamlit
        st.plotly_chart(fig_daily, use_container_width=True)
        st.plotly_chart(fig_hourly, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")



def generate_pdf_report(overview, daily, hourly, stats, recommendations):

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Sales Performance Report", ln=True, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, "\nExecutive Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "This report provides a comprehensive analysis of sales performance, including daily and hourly trends, payment status, and strategic recommendations.")

    # Iterate through sections dynamically
    for section_title, section_data in {
        "Financial Overview": overview,
        "Daily Performance": daily,
        "Hourly Performance": hourly,
        "Key Statistics": stats,
        "Recommendations": {str(i+1): rec for i, rec in enumerate(recommendations)}
    }.items():
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, f"\n{section_title}:", ln=True)
        pdf.set_font("Arial", size=12)
        for key, value in section_data.items():
            pdf.cell(100, 10, key, 0, 0)
            pdf.cell(100, 10, str(value), 0, 1)

    pdf_file = f"sales_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_file)
    return pdf_file


def analyze_sales(data):
    total_revenue = data['Total'].sum()
    total_payments = data['Paid'].sum()
    outstanding = total_revenue - total_payments
    payment_ratio = (total_payments / total_revenue * 100) if total_revenue > 0 else 0

    daily_sales = data.groupby(data['Date'].dt.date)['Total'].sum()
    best_day = daily_sales.idxmax()
    worst_day = daily_sales.idxmin()
    daily_growth = daily_sales.pct_change().mean() * 100

    hourly_sales = data.groupby('Hour')['Total'].sum()
    peak_hour = hourly_sales.idxmax()
    off_peak_hour = hourly_sales.idxmin()
    
    descriptive_stats = {
        " Average Daily Sales": f"${data.groupby('Date')['Total'].sum().mean():,.2f}",
        "Sales Volatility": f"{daily_sales.std():,.2f}",
        "Payment Ratio": f"{payment_ratio:.1f}%",
        "Daily Growth Rate": f"{daily_growth:.1f}%"
    }

    overview = {
        "Total Revenue": f"${total_revenue:,.2f}",
        "Total Payments": f"${total_payments:,.2f}",
        "Outstanding": f"${outstanding:,.2f}",
        "Payment Ratio": f"{payment_ratio:.1f}%"
    }

    daily_results = {
        "Best Day": f"{best_day} (${daily_sales[best_day]:,.2f})",
        "Worst Day": f"{worst_day} (${daily_sales[worst_day]:,.2f})",
        "Average Daily Sales": f"${daily_sales.mean():,.2f}",
        "Daily Growth Rate": f"{daily_growth:.1f}%"
    }

    hourly_results = {
        "Peak Hour": f"{peak_hour}:00 (${hourly_sales[peak_hour]:,.2f})",
        "Off-Peak Hour": f"{off_peak_hour}:00 (${hourly_sales[off_peak_hour]:,.2f})",
        "Peak/Off-Peak Ratio": f"{(hourly_sales[peak_hour] / hourly_sales[off_peak_hour]):,.2f}x"
    }

    return overview, daily_results, hourly_results, descriptive_stats

def generate_recommendations(data, hourly_results, daily_results):
    recommendations = []
    
    peak_hour = int(hourly_results["Peak Hour"].split(":")[0])
    recommendations.append(
        f"Optimize staffing levels during peak hours (around {peak_hour}:00) "
        "to maintain service quality and maximize sales potential."
    )
    
    payment_ratio = float(data['Paid'].sum() / data['Total'].sum() * 100)
    if payment_ratio < 95:
        recommendations.append(
            "Implement a more robust payment collection system to reduce outstanding payments. "
            "Consider offering early payment incentives."
        )
    
    daily_growth = float(daily_results["Daily Growth Rate"].strip('%'))
    if daily_growth < 0:
        recommendations.append(
            "Address declining daily sales trend by launching targeted marketing campaigns "
            "and reviewing pricing strategy."
        )
    
    recommendations.extend([
        "Develop promotional strategies for off-peak hours to boost sales during slower periods.",
        "Implement a customer loyalty program to encourage repeat business and increase average transaction value.",
        "Regular staff training to maintain service quality and increase sales effectiveness."
    ])
    
    return recommendations

def prepare_data_for_prediction(data):
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfMonth'] = df['Date'].dt.day
    
    df['Sales_Lag1'] = df.groupby('Hour')['Total'].shift(1)
    df['Sales_Lag7'] = df.groupby('Hour')['Total'].shift(7)
    
    df = df.dropna()
    
    return df

def train_prediction_model(data):
    features = ['Hour', 'DayOfWeek', 'Month', 'DayOfMonth', 'Sales_Lag1', 'Sales_Lag7']

    X = data[features]
    y = data['Total']

    # Ensure proper time-based train-test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Model with basic hyperparameter tuning
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)

    return model, X_test, y_test, test_preds, test_mae


def perform_customer_segmentation(data, n_clusters=4):
    if 'CustomerID' not in data.columns:
        raise ValueError("CustomerID column is required for segmentation analysis.")

    customer_metrics = data.groupby('CustomerID').agg({
        'Total': ['count', 'mean', 'sum'],
        'Date': lambda x: (x.max() - x.min()).days
    }).reset_index()

    customer_metrics.columns = ['CustomerID', 'Frequency', 'AvgPurchase', 'TotalSpent', 'DaysSinceFirst']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(customer_metrics.drop('CustomerID', axis=1))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_metrics['Segment'] = kmeans.fit_predict(features_scaled)

    segment_labels = {
        0: 'High Value',
        1: 'Medium Value',
        2: 'Low Value',
        3: 'New Customers'
    }
    customer_metrics['SegmentLabel'] = customer_metrics['Segment'].map(segment_labels)

    return customer_metrics


def create_forecast_indicator(current_value, predicted_value):
    percent_change = ((predicted_value - current_value) / current_value) * 100
    if percent_change > 0:
        return f"""
            <div class="indicator indicator-up">
                ↑ {abs(percent_change):.1f}% Predicted Increase
            </div>
        """
    else:
        return f"""
            <div class="indicator indicator-down">
                ↓ {abs(percent_change):.1f}% Predicted Decrease
            </div>
        """

def plot_segmentation_analysis(customer_metrics):
    fig1 = px.pie(customer_metrics, names='SegmentLabel', title='Customer Segment Distribution')
    fig2 = px.bar(customer_metrics.groupby('SegmentLabel')['AvgPurchase'].mean().reset_index(),
                  x='SegmentLabel', y='AvgPurchase',
                  title='Average Purchase Value by Segment')
    fig3 = px.bar(customer_metrics.groupby('SegmentLabel')['TotalSpent'].sum().reset_index(),
                  x='SegmentLabel', y='TotalSpent',
                  title='Total Revenue by Segment')
    
    return fig1, fig2, fig3

def plot_forecast_results(y_test, test_preds, test_dates):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=test_dates, y=y_test,
        name='Actual',
        mode='lines',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates, y=test_preds,
        name='Predicted',
        mode='lines',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Sales Forecast vs Actual',
        xaxis_title='Date',
        yaxis_title='Sales',
        showlegend=True
    )
    
    return fig

def main():
    st.title("📈 Sales Performance and Forecasting Tool")
    st.markdown("Upload your sales data to generate comprehensive insights and recommendations.")

    st.sidebar.header("Settings")
    prediction_days = st.sidebar.slider("Forecast Days", 7, 30, 14)
    segment_count = st.sidebar.slider("Number of Customer Segments", 3, 6, 4)
    confidence_interval = st.sidebar.slider("Prediction Confidence Interval", 80, 99, 95)

    uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            is_valid, message = validate_data(data)
            
            if not is_valid:
                st.error(message)
                return
                
            st.success("Data uploaded and validated successfully!")

            tabs = st.tabs([
                "Overview", 
                "Visualizations", 
                "Predictions", 
                "Customer Segmentation",
                "Detailed Analysis", 
                "Recommendations"
            ])
            
            overview, daily_results, hourly_results, stats = analyze_sales(data)
            recommendations = generate_recommendations(data, hourly_results, daily_results)

            with tabs[0]:
                st.header("Key Performance Metrics")
                create_metric_cards(overview)
                
                st.subheader("Recent Performance Trends")
                col1, col2, col3 = st.columns(3)
                with col1:
                    daily_change = float(daily_results["Daily Growth Rate"].strip('%'))
                    st.metric("Daily Growth", f"{daily_change:.1f}%", 
                             delta=f"{daily_change:.1f}%")
                with col2:
                    payment_ratio = float(overview["Payment Ratio"].strip('%'))
                    st.metric("Payment Collection", f"{payment_ratio:.1f}%", 
                             delta=f"{payment_ratio - 95:.1f}%")
                with col3:
                    peak_hour = hourly_results["Peak Hour"].split()[0]
                    st.metric("Peak Sales Hour", peak_hour)

            with tabs[1]:
                st.header("Sales Trends and Patterns")
                create_visualizations(data)
                
                st.subheader("Custom Date Range Analysis")
                date_range = st.date_input(
                    "Select Date Range",
                    value=(data['Date'].min(), data['Date'].max()),
                    key="date_range"
                )
                if len(date_range) == 2:
                    filtered_data = data[
                        (data['Date'] >= str(date_range[0])) & 
                        (data['Date'] <= str(date_range[1]))
                    ]
                    create_visualizations(filtered_data)

            with tabs[ 2]:
                st.header("Sales Predictions and Forecasting")
                
                prepared_data = prepare_data_for_prediction(data)
                model, X_test, y_test, test_preds, test_mae = train_prediction_model(prepared_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    current_avg = y_test.mean()
                    predicted_avg = test_preds.mean()
                    st.markdown("### Sales Trend Indicator")
                    st.markdown(
                        create_forecast_indicator(current_avg, predicted_avg), 
                        unsafe_allow_html=True
                    )
                
                with col2:
                    accuracy = (1 - test_mae/y_test.mean()) * 100
                    st.metric(
                        "Forecast Accuracy", 
                        f"{accuracy:.1f}%",
                        delta=f"{accuracy - 90:.1f}%"
                    )
                
                st.plotly_chart(
                    plot_forecast_results(y_test, test_preds, X_test.index),
                    use_container_width=True,
                    key="forecast_plot"
                )
                
                st.subheader("Prediction Factors")
                feature_importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    feature_importance, 
                    x='Feature', 
                    y='Importance',
                    title='Factors Influencing Sales Prediction'
                )
                st.plotly_chart(fig, use_container_width=True, key="feature_importance_plot")

            with tabs[3]:
                st.header("Customer Segmentation Analysis")
                
                if 'CustomerID' not in data.columns:
                    st.warning("""
                        CustomerID column is required for segmentation analysis. 
                        Please ensure your data includes customer identification.
                    """)
                else:
                    customer_metrics = perform_customer_segmentation(data)
                    
                    st.subheader("Customer Segments Overview")
                    segment_stats = customer_metrics.groupby('SegmentLabel').agg({
                        'CustomerID': 'count',
                        'TotalSpent': ['mean', 'sum'],
                        'Frequency': 'mean'
                    }).round(2)
                    
                    for segment in segment_stats.index:
                        stats = segment_stats.loc[segment]
                        st.markdown(f"""
                            <div class="segment-card">
                                <h3>{segment}</h3>
                                <p>Customer Count: {stats[('CustomerID', 'count')]:,}</p>
                                <p>Average Spent: ${stats[('TotalSpent', 'mean')]:,.2f}</p>
                                <p>Total Revenue: ${stats[('TotalSpent', 'sum')]:,.2f}</p>
                                <p>Average Visits: {stats[('Frequency', 'mean')]:.1f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    fig1, fig2, fig3 = plot_segmentation_analysis(customer_metrics)
                    st.plotly_chart(fig1, use_container_width=True, key="customer_segment_distribution")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig2, use_container_width=True, key="average_purchase_by_segment")
                    with col2:
                        st.plotly_chart(fig3, use_container_width=True, key="total_revenue_by_segment")

            with tabs[4]:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Daily Performance Metrics")
                    st.write(daily_results)
                    st.header("Hourly Performance Metrics")
                    st.write(hourly_results)
                with col2:
                    st.header("Statistical Insights")
                    st.write(stats)

            with tabs[5]:
                st.header("Strategic Recommendations")
                
                high_priority = []
                medium_priority = []
                low_priority = []
                
                for rec in recommendations:
                    if "urgent" in rec.lower() or "immediate" in rec.lower():
                        high_priority.append(rec)
                    elif "consider" in rec.lower() or "may" in rec.lower():
                        low_priority.append(rec)
                    else:
                        medium_priority.append(rec)
                
                with st.expander("High Priority Actions", expanded=True):
                    for i, rec in enumerate(high_priority, 1):
                        st.markdown(f"**{i}.** {rec}")
                
                with st.expander("Medium Priority Actions", expanded=True):
                    for i, rec in enumerate(medium_priority, 1):
                        st.markdown(f"**{i}.** {rec}")
                
                with st.expander("Low Priority Actions"):
                    for i, rec in enumerate(low_priority, 1):
                        st.markdown(f"**{i}.** {rec}")

            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                report_sections = st.multiselect(
                    "Select sections to include in the report",
                    ["Overview", "Daily Analysis", "Hourly Analysis", 
                     "Predictions", "Customer Segmentation", "Recommendations"],
                    default=["Overview", "Recommendations"]
                )
            
            with col2:
                if st.button("Generate PDF Report", type="primary"):
                    with st.spinner("Generating comprehensive PDF report..."):
                        pdf_file = generate_pdf_report(
                            overview, 
                            daily_results, 
                            hourly_results, 
                            stats, 
                            recommendations,
                            data
                        )
                        
                        with open(pdf_file, "rb") as f:
                            st.download_button(
                                "📥 Download Report",
                                f,
                                file_name=pdf_file,
                                mime="application/pdf",
                                help="Download the complete analysis report in PDF format"
                            )

        except Exception as e:
            st.error(f"An error occurred while processing the data: {str(e)}")
            st.info("""
                Please ensure your CSV file contains the required columns:
                - Date
                - Hour
                - Total
                - Paid
                - CustomerID (for segmentation)
            """)

if __name__ == "__main__":
    main()