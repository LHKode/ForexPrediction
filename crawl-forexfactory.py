# source: https://medium.com/@igorzabukovec/automate-web-crawling-with-selenium-python-part-1-85113660de96
import pandas as pd
import sys
from datetime import datetime, date, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

EXEC_PATH = "chromedriver.exe"

url = "https://www.forexfactory.com/calendar?week={target_week}"

def get_sunday(year):
    d = date(year, 1, 1)
    d += timedelta(days = 6 - d.weekday())
    while d.year == year and d - date.today() < timedelta(0):
        month = d.strftime("%b").lower()
        target_week = month + d.strftime("%d") + '.' + str(year)
        yield target_week
        d += timedelta(days=7)

def crawling(target_week, year):
    chrome_options = Options()
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument("--window-size=1920x1080")

    driver = webdriver.Chrome(chrome_options=chrome_options, executable_path=EXEC_PATH)

    # making request
    try:
        driver.get(url.format(target_week=target_week))
        # driver.execute_script('op.selectTimeZone(7);')
    except Exception as e:
        print('error in making request')
        return


    try:
        # add some delay before getting response (advice)

        # no. of cols:
        header = driver.find_elements_by_css_selector(".calendar__header--desktop")
        if len(header) == 1:
            cols = header[0].text.split()
        else:
            cols = []
        print(cols)
        num_cols = len(cols)

        # create DataFrame to save info
        df = pd.DataFrame(columns=['Date', 'Time', 'Currency', 'Impact', 'Event', 'Actual', 'Forecast', 'Previous'])

        #no. of rows:
        content = driver.find_elements_by_css_selector(".calendar__row:not(calendar__row--alt).calendar_row")
        for c in content:
            date = c.find_elements_by_css_selector(".date")[0].text
            if len(date.split()) > 0:
                date = ' '.join(date.split()) + ' {}'.format(year)
            else:
                date = None
            news_time = c.find_elements_by_css_selector(".time")[0].text
            currency = c.find_elements_by_css_selector(".currency")[0].text
            impact_class = c.find_elements_by_css_selector(".impact")[0].get_attribute("class")
            impact = 'No specified'
            if "low" in impact_class:
                impact = "Low"
            if "holiday" in impact_class:
                impact = "Holiday"
            if "high" in impact_class:
                impact = "High"
            if "medium" in impact_class:
                impact = "Medium"
            event = c.find_elements_by_css_selector(".event")[0].text
            actual = c.find_elements_by_css_selector(".actual")[0].text
            forecast = c.find_elements_by_css_selector(".forecast")[0].text
            previous = c.find_elements_by_css_selector(".previous")[0].text
            # print('Date: {date} Time: {news_time} Cur: {currency} Impact: {impact} Event: {event} Actual: {actual} Previous: {previous}'.format(\
            #     date=date,
            #     news_time=news_time,
            #     currency=currency,
            #     impact=impact,
            #     event=event,
            #     actual=actual,
            #     previous=previous
            # ))
            df.loc[df.shape[0]] = date, news_time, currency, impact, event, actual, forecast, previous
            # end for loop
        num_rows = len(content)

        # Summary
        # print('Table has {num_cols} cols and {num_rows} rows'.format(num_cols=num_cols, num_rows=num_rows))
        # print('Result shape {}'.format(df.shape[0]))
        # print('Info: ')
        # print(df.info())
        # print('Describe: ')
        # print(df.describe())
        # print('Head: ')
        # print(df.head(10))

        df.to_csv('{}_ff.csv'.format(target_week), index=False)
        driver.close()
        return

    except Exception as e:
        print(e)
        driver.close()
        return

if __name__ == '__main__':
    start_time = datetime.now()
    year = int(sys.argv[1])
    if len(sys.argv) > 2:
        from_year = int(sys.argv[2])
        print(x for x in range(from_year, year+1))
        for idx in range(from_year, year+1):
            allsundays = [d for d in get_sunday(idx)]
            for sunday in allsundays:
                print(sunday)
                crawling(sunday, idx)
    else:
        allsundays = [d for d in get_sunday(year)]
        for sunday in allsundays:
            print(sunday)
            crawling(sunday, year)
    print('Processing time: {}'.format(datetime.now() - start_time))
