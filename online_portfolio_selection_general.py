from abc import ABC, abstractmethod
import numpy as np


class StockPriceRelationsSubscriber(ABC):

    @abstractmethod
    def on_price_relations_updated(self, price_relations):
        pass


class PortfolioManager(StockPriceRelationsSubscriber):

    __periods = 0
    __current_period = 0

    def on_price_relations_updated(self, price_relations):
        print("Получены обновленные отношения цен: %s".format(price_relations))
        if self.__current_period > self.__periods:
            print("Завершаем выбор портфеля, достигунт последний торговый период")
            for producer in self.__price_relations_producers:
                producer.stop()

    def __init__(self, price_relations_producers, portfolio_selectors):
        self.__price_relations_producers = price_relations_producers
        self.__portfolio_selectors = portfolio_selectors
        self.__portfolios = dict()

    def start_auction(self, periods):
        self.__periods = periods
        for producer in self.__price_relations_producers:
            producer.register_subscribers(self.__portfolio_selectors)
            producer.register_subscriber(self)
            producer.start()


class OnlinePortfolioSelector(StockPriceRelationsSubscriber, ABC):

    def __init__(self):
        self.__subscribers = []

    def register_subscriber(self, subscriber):
        self.__subscribers.append(subscriber)


class StockPriceRelationsProducer(ABC):

    def __init__(self):
        self.__subscribers = []

    def register_subscriber(self, subscriber):
        self.__subscribers.append(subscriber)

    def register_subscribers(self, subscribers: list):
        self.__subscribers.extend(subscribers)

    def notify_subscribers(self, price_relation):
        for subscriber in self.__subscribers:
            subscriber.on_price_relations_updated(price_relation)

    @abstractmethod
    def start(self):
        pass

    def stop(self):
        self.__subscribers = []