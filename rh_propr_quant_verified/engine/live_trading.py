from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .broker_robinhood import RobinhoodCredentials, RobinhoodCryptoBroker


@dataclass
class LiveStatus:
    connected: bool
    message: str
    payload: Optional[dict] = None


class LiveTradingService:
    """Thin live-execution coordinator.

    This intentionally keeps live trading conservative: it validates credentials,
    supports account refresh, and submits manual orders only when explicitly asked.
    Strategy automation should stay disabled until the operator confirms API paths,
    symbols, and order schemas against official Robinhood documentation.
    """

    def __init__(self):
        self.broker: Optional[RobinhoodCryptoBroker] = None
        self.connected = False

    def configure(self, api_key: str, base64_private_key: str) -> None:
        creds = RobinhoodCredentials(api_key=api_key.strip(), base64_private_key=base64_private_key.strip())
        self.broker = RobinhoodCryptoBroker(creds)
        self.connected = False

    def connect(self) -> LiveStatus:
        if self.broker is None:
            return LiveStatus(False, 'credentials_not_loaded')
        try:
            payload = self.broker.get_account()
            self.connected = True
            return LiveStatus(True, 'account_refresh_ok', payload)
        except Exception as exc:
            self.connected = False
            return LiveStatus(False, f'connect_failed: {exc}')

    def refresh_account(self) -> LiveStatus:
        return self.connect()

    def submit_market_order(self, symbol: str, side: str, quote_amount: float) -> LiveStatus:
        if self.broker is None:
            return LiveStatus(False, 'credentials_not_loaded')
        try:
            payload = self.broker.place_market_order(symbol=symbol, side=side, quote_amount=quote_amount)
            return LiveStatus(True, 'market_order_submitted', payload)
        except Exception as exc:
            return LiveStatus(False, f'order_failed: {exc}')

    def submit_limit_order(self, symbol: str, side: str, limit_price: float, asset_quantity: float) -> LiveStatus:
        if self.broker is None:
            return LiveStatus(False, 'credentials_not_loaded')
        try:
            payload = self.broker.place_limit_order(symbol=symbol, side=side, limit_price=limit_price, asset_quantity=asset_quantity)
            return LiveStatus(True, 'limit_order_submitted', payload)
        except Exception as exc:
            return LiveStatus(False, f'order_failed: {exc}')

    def cancel_order(self, order_id: str) -> LiveStatus:
        if self.broker is None:
            return LiveStatus(False, 'credentials_not_loaded')
        try:
            payload = self.broker.cancel_order(order_id)
            return LiveStatus(True, 'order_cancel_requested', payload)
        except Exception as exc:
            return LiveStatus(False, f'cancel_failed: {exc}')
