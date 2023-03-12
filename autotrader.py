# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side

import math

LOT_SIZE = 20
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
ARBITRAGE_TICKS = 3
KAPPA = 2
GAMMA = 0.2
VOLATILITY = 0.05
TOTAL_TIME = 900
MAX_ORDER_TICKS = 3
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS


class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    class OrderBook():
        def __init__(self):
            self.ask_prices = []
            self.ask_volumes = []
            self.bid_prices = []
            self.bid_volumes = []

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        self.start_time = self.event_loop.time()
        self.etf_LOB = self.OrderBook()
        self.future_LOB = self.OrderBook()
        self.previous_order_book_sequence_number = 0
        # price, id, volume
        self.bid_orders = []
        self.ask_orders = []

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        # self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
        #                  price, volume)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        if self.previous_order_book_sequence_number > sequence_number:
            return
        self.previous_order_book_sequence_number = sequence_number
        # update local order book
        self.update_order_book(instrument, ask_prices, ask_volumes, bid_prices, bid_volumes)

        while True:
            # check the given prices
            if bid_prices[0] == 0 or ask_prices[0] == 0: break

            # only calculate price when having Future order book
            if instrument == Instrument.ETF: break

            # check if arbitrage opportunuity exist
            arb_done = False
            do_arb, arb_side = self.arbitrage_check()
            if do_arb:
                arb_done = self.arbitrage(arb_side)
            if arb_done: break

            # calculate price by AS model
            t = self.event_loop.time() - self.start_time
            time_frac = (TOTAL_TIME - t) / TOTAL_TIME

            mid_price = (bid_prices[0] + ask_prices[0]) / 2
            indiff_price = mid_price - self.position * GAMMA * (VOLATILITY ** 2) * time_frac
            # spread = 2/GAMMA + math.log(1 + gamma/KAPPA)
            spread = (GAMMA * (VOLATILITY ** 2) * time_frac + math.log(1 + GAMMA / KAPPA)) * TICK_SIZE_IN_CENTS

            new_bid_price = int(math.floor(((indiff_price - spread / 2) / TICK_SIZE_IN_CENTS)) * TICK_SIZE_IN_CENTS)
            new_ask_price = int(math.ceil(((indiff_price + spread / 2) / TICK_SIZE_IN_CENTS)) * TICK_SIZE_IN_CENTS)

            # check the quote
            if new_bid_price >= new_ask_price: break

            # go through history orders, cancel previous orders
            new_bid_price_low = new_bid_price - MAX_ORDER_TICKS * TICK_SIZE_IN_CENTS
            new_ask_price_high = new_ask_price + MAX_ORDER_TICKS * TICK_SIZE_IN_CENTS

            bid_pops_head = 0
            bid_pops_tail = 0
            for i in self.bid_orders:
                if i[0] > new_bid_price: bid_pops_head += 1
                if i[0] < new_bid_price_low: bid_pops_tail += 1
            self.bid_orders = self.pop_helper(self.bid_orders, bid_pops_head, bid_pops_tail)

            # fill the ordre list with possible price
            bid_price_list = [i for i in range(new_bid_price, new_bid_price_low, -TICK_SIZE_IN_CENTS)]
            if len(self.bid_orders) == 0:
                for i in bid_price_list:
                    self.bid_orders.append([i, 0, 0])
            else:
                price_stack = []
                for b_price in bid_price_list:
                    if b_price > self.bid_orders[0][0]:
                        price_stack.append(b_price)
                    if b_price < self.bid_orders[-1][0]:
                        for _ in range(len(price_stack)):
                            self.bid_orders.insert(0, [price_stack.pop(), 0, 0])
                        self.bid_orders.append([b_price, 0, 0])

            ask_pops_head = 0
            ask_pops_tail = 0
            for i in self.ask_orders:
                if i[0] < new_ask_price: ask_pops_head += 1
                if i[0] > new_ask_price_high: ask_pops_tail += 1
            self.ask_orders = self.pop_helper(self.ask_orders, ask_pops_head, ask_pops_tail)

            # fill the ordre list with possible price
            ask_price_list = [i for i in range(new_ask_price, new_ask_price_high, TICK_SIZE_IN_CENTS)]
            if len(self.ask_orders) == 0:
                for i in ask_price_list:
                    self.ask_orders.append([i, 0, 0])
            else:
                price_stack = []
                for a_price in ask_price_list:
                    if a_price < self.ask_orders[0][0]:
                        price_stack.append(a_price)
                    if a_price > self.ask_orders[-1][0]:
                        for _ in range(len(price_stack)):
                            self.ask_orders.insert(0, [price_stack.pop(), 0, 0])
                        self.ask_orders.append([a_price, 0, 0])

            # place order
            self.place_orders()
            break
        self.logger.info("received order book for instrument %d with sequence number %d, "
                         "Actual Position - %d, Real Position(Bid) - %d, Real Position(ASK) - %d", instrument,
                         sequence_number, self.position, self.get_real_position(Side.BID),
                         self.get_real_position(Side.ASK))

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        if client_order_id in self.bids:
            # update remaining volumes
            for i in self.bid_orders:
                if i[1] == client_order_id:
                    i[2] -= volume
                    break
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks:
            # update remaining volumes
            for i in self.ask_orders:
                if i[1] == client_order_id:
                    i[2] -= volume
                    break
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume)

        self.logger.info("received order filled for order %d with price %d and volume %d,"
                         "Actual Position - %d, Real Position(Bid) - %d, Real Position(ASK) - %d", client_order_id,
                         price, volume, self.position, self.get_real_position(Side.BID),
                         self.get_real_position(Side.ASK))

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        if remaining_volume == 0:
            if client_order_id in self.bids:
                self.bids.discard(client_order_id)
                for i, bid in enumerate(self.bid_orders):
                    if client_order_id == bid[1]:
                        del self.bid_orders[i]
                        break
            elif client_order_id in self.asks:
                self.asks.discard(client_order_id)
                for i, ask in enumerate(self.ask_orders):
                    if client_order_id == ask[1]:
                        del self.ask_orders[i]
                        break
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d,"
                         "Actual Position - %d, Real Position(Bid) - %d, Real Position(ASK) - %d",
                         client_order_id, fill_volume, remaining_volume, fees, self.position,
                         self.get_real_position(Side.BID), self.get_real_position(Side.ASK))

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)

    def update_order_book(self, instrument: int, ask_prices: List[int]
                          , ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        target = None
        if instrument == Instrument.ETF:
            target = self.etf_LOB
        elif instrument == Instrument.FUTURE:
            target = self.future_LOB

        target.ask_prices = ask_prices
        target.ask_volumes = ask_volumes
        target.bid_prices = bid_prices
        target.bid_volumes = bid_volumes

    def pop_helper(self, order_list: List, pop_head: int, pop_tail: int):
        for _ in range(pop_head):
            order = order_list.pop(0)
            self.send_cancel_order(order[1])

        for _ in range(pop_tail):
            order = order_list.pop(-1)
            self.send_cancel_order(order[1])

        return order_list

    def get_real_position(self, side):
        if side == Side.ASK:
            remain_pos = 0
            for i in self.ask_orders:
                remain_pos += i[2]
            return self.position - remain_pos

        elif side == Side.BID:
            remain_pos = 0
            for i in self.bid_orders:
                remain_pos += i[2]
            return self.position + remain_pos

    def arbitrage_check(self):
        if len(self.etf_LOB.bid_prices) == 0 or len(self.future_LOB.bid_prices) == 0:
            return False, None
        etf_bid_price = self.etf_LOB.ask_prices[0]
        etf_ask_price = self.etf_LOB.bid_prices[0]
        future_bid_price = self.future_LOB.ask_prices[0]
        future_ask_price = self.future_LOB.bid_prices[0]

        if future_ask_price - etf_bid_price > TICK_SIZE_IN_CENTS * ARBITRAGE_TICKS:
            return True, Side.BID

        elif etf_ask_price - future_bid_price > TICK_SIZE_IN_CENTS * ARBITRAGE_TICKS:
            return True, Side.ASK
        else:
            return False, None

    def arbitrage(self, side):

        if side == Side.BID:
            # cancel all ask orders
            arb_bid_price = self.etf_LOB.ask_prices[0]
            for i in self.ask_orders:
                if arb_bid_price <= i[0]:
                    self.send_cancel_order(i[1])

            avaliable_pos = min(LOT_SIZE, POSITION_LIMIT - self.get_real_position(Side.BID),
                                self.etf_LOB.ask_volumes[0])
            if avaliable_pos <= 0: return False

            order_id = next(self.order_ids)
            self.send_insert_order(order_id, Side.BUY, arb_bid_price, avaliable_pos, Lifespan.FILL_AND_KILL)
            self.bids.add(order_id)

        elif side == Side.ASK:
            # cancel all bid orders
            arb_ask_price = self.etf_LOB.bid_prices[0]
            for i in self.bid_orders:
                if arb_ask_price <= i[1]:
                    self.send_cancel_order(i[1])

            avaliable_pos = min(LOT_SIZE, POSITION_LIMIT + self.get_real_position(Side.ASK),
                                self.etf_LOB.bid_volumes[0])
            if avaliable_pos <= 0: return False

            order_id = next(self.order_ids)
            self.send_insert_order(order_id, Side.SELL, arb_ask_price, avaliable_pos, Lifespan.FILL_AND_KILL)
            self.asks.add(order_id)

        return True

    def place_orders(self):
        for i in self.bid_orders:
            if i[1] == 0 and self.get_real_position(Side.BID) < POSITION_LIMIT:
                avaliable_lot = min(LOT_SIZE, POSITION_LIMIT - self.get_real_position(Side.BID))
                if avaliable_lot == 0:
                    break
                else:
                    i[1] = next(self.order_ids)
                    i[2] = avaliable_lot
                    self.send_insert_order(i[1], Side.BUY, i[0], avaliable_lot, Lifespan.GOOD_FOR_DAY)
                    self.bids.add(i[1])

        for i in self.ask_orders:
            if i[1] == 0 and self.get_real_position(Side.ASK) > -POSITION_LIMIT:
                avaliable_lot = min(LOT_SIZE, POSITION_LIMIT + self.get_real_position(Side.ASK))
                if avaliable_lot == 0:
                    break
                else:
                    i[1] = next(self.order_ids)
                    i[2] = avaliable_lot
                    self.send_insert_order(i[1], Side.SELL, i[0], avaliable_lot, Lifespan.GOOD_FOR_DAY)
                    self.asks.add(i[1])
