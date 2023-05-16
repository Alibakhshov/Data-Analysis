import telebot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Set the matplotlib backend to Agg
plt.switch_backend('agg')

# Telegram Bot Token
TOKEN = '5806370469:AAGNIWuqXcCgBInVQSPdJMJch0CuRXPJaiQ'

# Load the dataset
dataset_path = 'house_prices_small.csv'
df = pd.read_csv(dataset_path)

# Create a linear regression model
regression_model = LinearRegression()

# Create a scatterplot function
def draw_scatterplot(feature):  
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    sns.scatterplot(x=df[feature], y=df['SalePrice'], ax=ax)
    ax.set_xlabel(feature)
    ax.set_ylabel('SalePrice')
    ax.set_title(f'{feature} vs. SalePrice')
    canvas.draw()
    fig.savefig('scatterplot.png', dpi=150, bbox_inches='tight')  # Save the scatterplot as an image file


# Function to draw scatterplot of residuals
def draw_residuals_scatterplot(feature):
    model = regression_models[feature]
    X = df[[feature]]
    y = df['SalePrice']
    y_pred = model.predict(X)
    residuals = y - y_pred

    plt.figure()
    sns.scatterplot(x=y_pred, y=residuals)
    plt.xlabel('Predicted SalePrice')
    plt.ylabel('Residuals')
    plt.title(f'Scatterplot of Residuals ({feature})')

    # Save the scatterplot as an image file
    plt.savefig('residuals_plot.png', dpi=150, bbox_inches='tight')
    
# Function to generate a report for the regression model
def generate_model_report(feature):
    model = regression_models[feature]
    X = df[[feature]]
    y = df['SalePrice']
    y_pred = model.predict(X)
    residuals = y - y_pred
    r2 = r2_scores[feature]

    report = f"Regression Model Report ({feature}):\n\n"
    report += f"R-squared: {r2}\n\n"
    report += f"Model Coefficients: {model.coef_}\n\n"
    report += f"Intercept: {model.intercept_}\n\n"
    report += f"Sample Predictions:\n"
    for i in range(5):
        report += f"  - Actual: {y[i]}, Predicted: {y_pred[i]}, Residual: {residuals[i]}\n"

    return report

# Function to identify and remove outliers, and generate reports without outliers
def remove_outliers_and_generate_reports(feature):
    model = regression_models[feature]
    X = df[[feature]]
    y = df['SalePrice']

    # Fit the model with all data
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    r2_with_outliers = r2_scores[feature]

    # Detect and remove outliers
    outliers_indices = detect_outliers(residuals)
    X_filtered = X.drop(outliers_indices)
    y_filtered = y.drop(outliers_indices)

    # Refit the model without outliers
    model.fit(X_filtered, y_filtered)
    y_pred_filtered = model.predict(X_filtered)
    residuals_filtered = y_filtered - y_pred_filtered
    r2_without_outliers = r2_score(y_filtered, y_pred_filtered)

    # Generate reports
    scatterplot_path = f"{feature}_scatterplot.png"
    residuals_scatterplot_path = f"{feature}_residuals_scatterplot.png"
    report = generate_model_report(feature)

    # Save scatterplot without outliers
    plt.figure()
    sns.scatterplot(x=X_filtered.values.flatten(), y=y_filtered.values.flatten())
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(f'Scatterplot without Outliers ({feature})')
    plt.savefig(scatterplot_path, dpi=150, bbox_inches='tight')

    # Save scatterplot of residuals without outliers
    plt.figure()
    sns.scatterplot(x=y_pred_filtered, y=residuals_filtered)
    plt.xlabel('Predicted SalePrice')
    plt.ylabel('Residuals')
    plt.title(f'Scatterplot of Residuals without Outliers ({feature})')
    plt.savefig(residuals_scatterplot_path, dpi=150, bbox_inches='tight')

    return r2_with_outliers, r2_without_outliers, scatterplot_path, residuals_scatterplot_path, report

# Function to detect outliers using z-score method
def detect_outliers(data):
    z_scores = (data - np.mean(data)) / np.std(data)
    outliers_indices = np.where(np.abs(z_scores) > 3)[0]
    return outliers_indices
    
# Create a linear regression model for each numerical feature
regression_models = {
    'LotArea': LinearRegression(),
    'OverallQual': LinearRegression(),
    'YearBuilt': LinearRegression()
}

# Fit the regression models and calculate squared R values
r2_scores = {}
for feature, model in regression_models.items():
    X = df[[feature]]
    y = df['SalePrice']
    model.fit(X, y)
    y_pred = model.predict(X)
    r2_scores[feature] = r2_score(y, y_pred)

# Create the Telegram bot
bot = telebot.TeleBot(TOKEN)

# Handle the /squaredr command
@bot.message_handler(commands=['squaredr'])
def send_squared_r(message):
    try:
        response = "Squared R values:\n\n"
        for feature, r2_score_value in r2_scores.items():
            response += f"{feature}: {r2_score_value}\n"
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")

# Handle the /start command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Welcome to the House Price Regression Bot!")
    
# Handle the /residuals command
@bot.message_handler(commands=['residuals'])
def send_residuals_scatterplot(message):
    try:
        draw_residuals_scatterplot('LotArea')
        bot.send_photo(message.chat.id, open('residuals_plot.png', 'rb'))
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")
        
# Handle the /report command
@bot.message_handler(commands=['report'])
def send_model_report(message):
    try:
        report = generate_model_report('LotArea')
        bot.reply_to(message, report)
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")

# Handle the /scatterplot command
@bot.message_handler(commands=['scatterplot'])
def send_scatterplot(message):
    try:
        draw_scatterplot(feature='LotArea')
        bot.send_photo(message.chat.id, open('scatterplot.png', 'rb'))
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")

# Handle the /removeoutliers command
@bot.message_handler(commands=['removeoutliers'])
def send_reports_without_outliers(message):
    try:
        r2_with_outliers, r2_without_outliers, scatterplot_path, residuals_scatterplot_path, report = \
            remove_outliers_and_generate_reports('LotArea')

        bot.reply_to(message, f"Squared R with outliers: {r2_with_outliers}\nSquared R without outliers: {r2_without_outliers}")
        bot.send_photo(message.chat.id, open(scatterplot_path, 'rb'))
        bot.send_photo(message.chat.id, open(residuals_scatterplot_path, 'rb'))
        bot.reply_to(message, report)
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")


# Handle any other text message
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    bot.reply_to(message, "I'm sorry, I didn't understand that command.")

# Run the bot
bot.polling()
