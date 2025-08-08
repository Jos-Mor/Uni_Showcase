# Water Alert Scraper

A simple Python script that monitors my local water company's website and sends Discord notifications when water service interruptions are announced.

## What it does

- Scrapes the municipal water service announcements page every 6 hours
- Looks for keywords related to water cuts/maintenance
- Sends a Discord message if something relevant is found
- Keeps track of the last message seen to avoid spam

## Why I built it

Got tired of manually checking the website to see if my water was going to be cut off. This runs automatically and gives me a heads up.

## Tech used

- Python with `requests` and `BeautifulSoup` for scraping
- Discord webhooks for notifications
- Basic file I/O for state tracking
- Crontab for automatic running

Pretty straightforward automation script - nothing fancy, just solves a real problem I had.
Unfortunately don't have a Raspberry Pi that I can keep on 24/7 so that's why there's the redundancy of crontab + time checking on the script itself.
