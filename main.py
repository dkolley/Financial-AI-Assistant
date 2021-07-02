from neuralintents import GenericAssistant
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import mplfinance as mpf
import yfinance as yf
import random
from textblob import TextBlob
import tweepy
import datetime
from datetime import date
from datetime import timedelta
from prettytable import PrettyTable
from reuterspy import Reuters
from yahoo_fin import stock_info as si
import requests
import pickle
import sys
import datetime as dt


def plot_rsi():
    ticker = str(input("Enter a ticker to see the RSI chart for it: ")).upper()
    start = dt.datetime(2018, 1, 1)
    end = dt.datetime.now()
    data = web.DataReader(ticker, 'yahoo', start, end)
    delta = data['Adj Close'].diff(1)
    delta.dropna(inplace=True)
    positive = delta.copy()
    negative = delta.copy()
    positive[positive < 0] = 0
    negative[negative > 0] = 0
    days = 14
    average_gain = positive.rolling(window=days).mean()
    average_loss = abs(negative.rolling(window=days).mean())
    relative_strength = average_gain/average_loss
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))
    combined = pd.DataFrame()
    combined['Adj Close'] = data['Adj Close']
    combined['RSI'] = RSI
    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(211)
    ax1.plot(combined.index, combined['Adj Close'], color='lightgray')
    ax1.set_title("Adjusted Close Price", color='white')
    ax1.grid(True, color='#555555')
    ax1.set_axisbelow(True)
    ax1.set_facecolor('black')
    ax1.figure.set_facecolor('#121212')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(combined.index, combined['RSI'], color='lightgray')
    ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.set_title(f"RSI Value: {ticker}", color='white')
    ax2.grid(False)
    ax2.set_axisbelow(True)
    ax2.set_facecolor('black')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    plt.show()


def calculate_intrinsic_value():

    try:
        stock_name = input("Enter the ticker you wish to find the intrinsic value for: ").lower()
        free_cash_flow_2020 = int(input("Enter the free cash flow value for 2020 (in thousands): "))
        free_cash_flow_2019 = int(input("Enter the free cash flow value for 2019 (in thousands): "))
        free_cash_flow_2018 = int(input("Enter the free cash flow value for 2018 (in thousands): "))
        free_cash_flow_2017 = int(input("Enter the free cash flow value for 2017 (in thousands): "))

        tckr = yf.Ticker(stock_name)
        outstanding_shares = tckr.info['sharesOutstanding']

        #constants
        required_rate = 0.07
        perpetual_rate = 0.02
        cash_flow_growth_rate = 0.03

        years = [1, 2, 3, 4]
        free_cash_flow = [free_cash_flow_2017, free_cash_flow_2018, free_cash_flow_2019, free_cash_flow_2020]
        future_free_cash_flow = []
        discount_factor = []
        discounted_future_free_cash_flow = []

        terminal_value = free_cash_flow[-1] * (1+perpetual_rate)/(required_rate-perpetual_rate)
        for year in years:
            cash_flow = free_cash_flow[-1] * (1+cash_flow_growth_rate)**year
            future_free_cash_flow.append(cash_flow)
            discount_factor.append((1+required_rate)**year)
        for i in range(0, len(years)):
            discounted_future_free_cash_flow.append(future_free_cash_flow[i]/discount_factor[i])
        discounted_terminal_value = terminal_value/(1 + required_rate)**4
        discounted_future_free_cash_flow.append(discounted_terminal_value)
        value_today = sum(discounted_future_free_cash_flow)
        fair_value = value_today*1000/outstanding_shares

        print("The current intrinsic value of " + stock_name.upper() + " is ${}".format(round(fair_value, 2)))
    except:
        print("Enter a ticker that exists and ensure you submit values with no commas in them!")

with open('portfolio.pkl', 'rb') as f:
    portfolio = pickle.load(f)


def save_portfolio():
    with open('portfolio.pkl', 'wb') as f:
        pickle.dump(portfolio, f)


def add_portfolio():
    try:
        ticker = input("Which stock do you want to add: ").upper()
        data = web.DataReader(ticker, 'yahoo')
        amount = input("How many shares do you want to add: ")
        if ticker in portfolio.keys():
            portfolio[ticker] += int(amount)
        else:
            portfolio[ticker] = int(amount)
        print(f"I have successfully added {amount} shares of {ticker} to your portfolio!")
        print()
        print("Your updated portfolio:")
        show_portfolio()
        save_portfolio()
    except:
        print(f"I do not recognize the stock under the ticker of {ticker}")
        print("Stock was not added!")


def remove_portfolio():
    ticker = input("Which stock do you want to sell: ").upper()
    amount = input("How many shares do you want to sell: ")
    if ticker in portfolio.keys():
        if int(amount) <= int(portfolio[ticker]):
            portfolio[ticker] -= int(amount)
            save_portfolio()
            print(f"I have successfully removed {amount} shares of {ticker}")
            print()
            print("Your updated portfolio:")
            show_portfolio()
        else:
            print("You don't have enough shares!")
    else:
        print(f"You don't own any shares of {ticker}")
    for stocksymbol in list(portfolio):
        if (portfolio[stocksymbol] == 0):
            del portfolio[stocksymbol]


def show_portfolio():
    print("Your portfolio:")
    for ticker in portfolio.keys():
        if (not (portfolio[ticker] == 0)):
            print(f"You own {portfolio[ticker]} shares of {ticker}")


def portfolio_worth():
    sum = 0
    for ticker in portfolio.keys():
        data = web.DataReader(ticker, 'yahoo')
        price = data['Close'].iloc[-1]
        sum += float(price) * float(portfolio[ticker])
    sum = str(sum)
    currIndex = sum.index(".")
    sum = sum[:currIndex + 3]
    print(f"Your portfolio is worth {sum} USD")


def portfolio_gains():
    starting_date = input("Enter a date for comparison (YYYY-MM-DD)")
    sum_now = 0
    sum_then = 0
    try:
        for ticker in portfolio.keys():
            data = web.DataReader(ticker, 'yahoo')
            price_now = data['Close'].iloc[-1]
            price_then = data.loc[data.index == starting_date]['Close'].values[0]
            sum_now += price_now
            sum_then += price_then
        print(f"Relative Gains: {((sum_now - sum_then) / sum_then) * 100}%")
        print(f"Absolute Gains: {sum_now - sum_then} USD")
    except IndexError:
        print("There was no trading on this day!")


def plot_chart():
    ticker = input("Choose a ticker symbol: ")
    starting_string = input("Choose a starting date (DD/MM/YYYY): ")
    plt.style.use('dark_background')
    start = dt.datetime.strptime(starting_string, "%d/%m/%Y")
    end = dt.datetime.now()

    data = web.DataReader(ticker, 'yahoo', start, end)
    colors = mpf.make_marketcolors(up='#00ff00', down='#ff0000', wick='inherit', edge='inherit', volume='in')
    mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=colors)
    mpf.plot(data, type='candle', style=mpf_style, volume=True)


def bye():
    print("Have a nice day; I hope I was of help for you today. Goodbye!")
    sys.exit(0)


def stock_price():
    ticker = input("Choose a ticker symbol: ")
    try:
        data = web.DataReader(ticker, 'yahoo')
        price = data['Close'].iloc[-1]
        price = str(price)
        currIndex = price.index(".")
        price = price[:currIndex + 3]
        print(f"The current price of {ticker} is ${price}")
    except:
        print("Sorry I do not recognize that ticker")


def get_specific_stock_price(ticker_name):
    data = web.DataReader(ticker_name, 'yahoo')
    price = data['Close'].iloc[-1]
    price = str(price)
    currIndex = price.index(".")
    price = price[:currIndex + 3]
    return price


def sma_of_stock():
    ticker_name = input("Enter the ticker name: ")
    period = input("Enter the period for the SMA (in days): ")
    api_keys = '#####################'
    yesterday = date.today() - datetime.timedelta(days=1)
    ti = TechIndicators(key=api_keys, output_format='pandas')
    try:
        data_ti = ti.get_sma(symbol=str(ticker_name), interval='daily', time_period=int(period), series_type='close')
        myStr = (data_ti[0])
        myStr = str(myStr)
        myStr = str(myStr.splitlines()[-3:-2])
        indexSpace = myStr.index(" ")
        myStr = myStr[indexSpace:].strip()
        myStr = myStr[:len(myStr) - 2]
        myStr = str(myStr)
        currIndex = myStr.index(".")
        myStr = myStr[:currIndex + 3]
        print(f"The simple moving average of the ticker {ticker_name.upper()} based on a period of {period} days is: ")
        print("$" + myStr)
    except:
        print(f"I'm sorry. I do not recognize the ticker {ticker_name.upper()}.")


def get_sentiment():
    api_key = '########################'
    api_key_secret = '########################'
    access_token = '########################'
    access_token_secret = 'kG4F1nUp8saNLWaTOjaOgMm9vbjG5rMFqhyr9w8u5sGIp'

    auth_handler = tweepy.OAuthHandler(consumer_key=api_key, consumer_secret=api_key_secret)
    auth_handler.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth_handler, wait_on_rate_limit=True)

    keyword = input("Enter a keyword you wish to get a sentiment rating for: ")
    search_term = keyword
    tweet_amount = 200

    polarity = 0
    positive = 0
    neutral = 0
    negative = 0

    tweets = tweepy.Cursor(api.search, q=search_term, lang='en').items(tweet_amount)
    for tweet in tweets:
        final_text = tweet.text.replace('RT', '')
        if final_text.startswith(' @'):
            position = final_text.index(':')
            final_text = final_text[position + 2:]
        if final_text.startswith('@'):
            position = final_text.index(' ')
            final_text = final_text[position + 2:]

        analysis = TextBlob(final_text)
        tweet_polarity = analysis.polarity
        if tweet_polarity > 0:
            positive += 1
        elif tweet_polarity < 0:
            negative += 1
        else:
            neutral += 1
        polarity += tweet_polarity

    print(f"For the keyword: {search_term}")
    print(f"Total Polarity: {polarity}")
    print(f"Average Polarity: {polarity / tweet_amount}")
    print(f"Amount of positive tweets: {positive}")
    print(f"Amount of negative tweets: {negative}")
    print(f"Amount of neutral tweets: {neutral}")


def sma_of_stock_screener(stock_name, period):
    api_keys = '###################'
    yesterday = date.today() - datetime.timedelta(days=1)
    ti = TechIndicators(key=api_keys, output_format='pandas')
    data_ti = ti.get_sma(symbol=str(stock_name), interval='daily', time_period=int(period), series_type='close')
    myStr = (data_ti[0])
    myStr = str(myStr)
    myStr = str(myStr.splitlines()[-3:-2])
    indexSpace = myStr.index(" ")
    myStr = myStr[indexSpace:].strip()
    myStr = myStr[:len(myStr) - 2]
    myStr = str(myStr)
    currIndex = myStr.index(".")
    myStr = myStr[:currIndex + 3]
    return myStr


def portfolio_visualizer():
    tickers = []
    amounts = []
    prices = []
    total = []

    for i in portfolio.keys():
        if(not(portfolio[i] == 0)):
            tickers.append(str(i.upper()))
            amounts.append(int(portfolio[i]))

    for ticker in tickers:
        df = web.DataReader(ticker, 'yahoo', dt.datetime(2019,8,1), dt.datetime.now())
        price = df[-1:]['Close'][0]
        prices.append(price)
        index = tickers.index(ticker)
        total.append(price * amounts[index])

    fig, ax = plt.subplots(figsize=(16,8))

    ax.set_facecolor('black')
    ax.figure.set_facecolor('#121212')

    ax.tick_params(axis='x', color='white')
    ax.tick_params(axis='y', color='white')

    ax.set_title('Dennis Kolley\'s Portfolio Visualizer', color="#EF6C35", fontsize=20)
    _, texts, _ = ax.pie(total, labels=tickers, autopct="%1.1f%%", pctdistance=0.8)
    [text.set_color('white') for text in texts]
    my_circle = plt.Circle((0,0), 0.55, color='black')
    plt.gca().add_artist(my_circle)
    ax.text(-2, 1, 'PORTFOLIO OVERVIEW', fontsize=14, color='#FFE536', verticalalignment='center', horizontalalignment='center')
    ax.text(-2, 0.85, f'Total USD Amount: {sum(total):.2f} $', fontsize=12, color="white", verticalalignment='center', horizontalalignment='center')
    counter = 0.15
    for ticker in tickers:
        ax.text(-2, 0.85 - counter, f'{ticker}: {total[tickers.index(ticker)]:.2f} $', fontsize=12, color="white", verticalalignment="center", horizontalalignment='center')
        counter += 0.15
    plt.show()

def help():
    print("You can ask me to do multiple tasks, such as: ")
    print()
    print("For a specific stock, I can: ")
    print("--Plot an RSI chart (to evaluate overbought or oversold conditions in a particular stock)")
    print("--Calculate the intrinsic value (to help determine the true value of an asset, mainly used by value investors)")
    print("--Plot the chart of it (to track the stock's current and historical price action)")
    print("--Give you the SMA of a stock over a certain time period")
    print("--Give you the real time price of the stock")
    print()
    print("I can also: ")
    print("--Store stocks in your personal portfolio")
    print("--Remove or add stocks from this portfolio")
    print("--Show your current portfolio, its current worth and its gains")
    print("--Display portfolio as a pie chart using a portfolio vizualizer")
    print("--Display all the stocks in your portfolio as a stock screener (to get in depth info about current stocks e.g current price vs 200SMA")
    print("--Get a sentiment result for a specific keyword using tweets from Twitter")
    print()
    print()


def stock_screener():
    print()
    print("Please wait while we generate your stock screener.......")
    print()
    tickers = si.tickers_sp500()
    start = dt.datetime.now() - dt.timedelta(days=365)
    end = dt.datetime.now()
    sp500_df = web.DataReader('^GSPC', 'yahoo', start, end)
    sp500_df['Pct Change'] = sp500_df['Adj Close'].pct_change()
    sp500_return = (sp500_df['Pct Change'] + 1).cumprod()[-1]
    return_list = []


    for ticker in portfolio.keys():
        df = web.DataReader(ticker, 'yahoo', start, end)
        df.to_csv(f'stock_data/{ticker}.csv')
        df['Pct Change'] = df['Adj Close'].pct_change()
        stock_return = (df['Pct Change'] + 1).cumprod()[-1]
        returns_compared = round((stock_return / sp500_return), 2)
        return_list.append(returns_compared)

    t = PrettyTable(
        ['Symbol', 'Current Price', 'SMA_72', 'SMA_150', 'SMA_200', 'Price vs 200-Day', '% Off High', 'PE Ratio',
         '52 Week Low', '52 Week High'])

    for stock in portfolio.keys():
        if not(portfolio[stock] == 0):
            df = pd.read_csv(f'stock_data/{stock}.csv', index_col=0)
            moving_averages = [72, 150, 200]
            for ma in moving_averages:
                df['SMA_' + str(ma)] = round(df['Adj Close'].rolling(window=ma).mean(), 2)


            price = get_specific_stock_price(stock)
            sma_seventytwo = df['SMA_72'][-1]
            sma_onefifty = df['SMA_150'][-1]
            sma_twohundred = df['SMA_200'][-1]
            diff_200sma_and_current_temp = float(price) - float(sma_twohundred)
            diff_200sma_and_current = round(diff_200sma_and_current_temp, 1)
            pe_ratio = float(si.get_quote_table(stock)['PE Ratio (TTM)'])
            peg_ratio = float(si.get_stats_valuation(stock)[1][4])
            low_52week = round(min(df['Low'][-(52 * 5):]), 2)
            high_52week = round(max(df['High'][-(52 * 5):]), 2)
            percent_off_high = round(100 * ((float(price) - float(high_52week)) / float(high_52week)), 2)
            t.add_row(
                [stock.upper(), price, sma_seventytwo, sma_onefifty, sma_twohundred, diff_200sma_and_current, percent_off_high, pe_ratio,
                 low_52week, high_52week])
    print(t)


mappings = {
    'plot_chart': plot_chart,
    'add_portfolio': add_portfolio,
    'remove_portfolio': remove_portfolio,
    'show_portfolio': show_portfolio,
    'portfolio_worth': portfolio_worth,
    'portfolio_gains': portfolio_gains,
    'stock_price': stock_price,
    'bye': bye,
    'sma_of_stock': sma_of_stock,
    'get_sentiment': get_sentiment,
    'stock_screener': stock_screener,
    'portfolio_visualizer':portfolio_visualizer,
    'plot_rsi':plot_rsi,
    'calculate_intrinsic_value':calculate_intrinsic_value,
    'help':help
}

assistant = GenericAssistant('intents.json', mappings, "financial_assistant_model")
assistant.train_model()
assistant.save_model()

counterOfResponses = 0

while True:
    print()
    if (counterOfResponses == 0):
        print("Welcome back Dennis!")
        print("What can I help you with today?")
    else:
        randNum = random.randint(1, 3)
        if (randNum == 1):
            print("Anything else?")
        elif (randNum == 2):
            print("Glad I could be of help. Need anything else?: ")
        else:
            print("What else can I do for you today?: ")
    message = input("")
    assistant.request(message)
    counterOfResponses += 1
