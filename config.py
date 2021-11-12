import os

PRODUCTION = "production"
DEVELOPMENT = "development"

COIN_TARGET = "BNB"
COIN_REFER = "USDT"

ENV = os.getenv("ENVIRONMENT", PRODUCTION)
DEBUG = False

# futures
BINANCE = {
  "key": "9c5a3bdbe030a794a0b4920a7916bdaf0c5c650af8aac7e094fdeeef9a2eae08",
  "secret": "f2e9e9229556b0d9258c807930fc3e61d5937ba3db011b56929b2a0cb20274b6"
}

# spot
# BINANCE = {
#   "key": "DA9Rm9HKVsBQ8hXjbj6omu1vY7ZaAPWDZ8sF01lN89ih4AfVh629KqfLQa2UO4w5",
#   "secret": "VHfl78kWdS6VqPhhoh7S8BXyhzcDIwZixNDoFfNdJ9U6PhpKbUeWSpsCIlTbhh9v"
# }

TELEGRAM = {
  "channel": "<CHANEL ID>",
  "bot": "<BOT KEY HERE>"
}

print("ENV = ", ENV)