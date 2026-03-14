from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


@dataclass
class RobinhoodCredentials:
    api_key: str
    base64_private_key: str
    base_url: str = "https://trading.robinhood.com"


class RobinhoodCryptoBroker:
    """Robinhood Crypto API adapter.

    Important: exact endpoints and request schemas must be validated against the
    latest official Robinhood documentation before use with real funds.
    """

    def __init__(self, creds: RobinhoodCredentials):
        self.creds = creds
        self.session = requests.Session()

    def _private_key(self) -> Ed25519PrivateKey:
        raw = base64.b64decode(self.creds.base64_private_key)
        return Ed25519PrivateKey.from_private_bytes(raw)

    def _headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time()))
        message = f"{self.creds.api_key}{timestamp}{path}{method}{body}".encode()
        signature = self._private_key().sign(message)
        return {
            "x-api-key": self.creds.api_key,
            "x-signature": base64.b64encode(signature).decode(),
            "x-timestamp": timestamp,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = json.dumps(payload) if payload else ""
        url = self.creds.base_url + path
        response = self.session.request(method, url, headers=self._headers(method, path, body), data=body or None, timeout=20)
        response.raise_for_status()
        if not response.text:
            return {}
        return response.json()

    def get_account(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/crypto/trading/accounts/")

    def get_best_bid_ask(self, symbol: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}")

    def place_market_order(self, symbol: str, side: str, quote_amount: float) -> Dict[str, Any]:
        payload = {
            "client_order_id": str(uuid.uuid4()),
            "symbol": symbol,
            "side": side,
            "type": "market",
            "quote_amount": f"{quote_amount:.8f}",
        }
        return self._request("POST", "/api/v1/crypto/trading/orders/", payload)

    def place_limit_order(self, symbol: str, side: str, limit_price: float, asset_quantity: float) -> Dict[str, Any]:
        payload = {
            "client_order_id": str(uuid.uuid4()),
            "symbol": symbol,
            "side": side,
            "type": "limit",
            "limit_price": f"{limit_price:.8f}",
            "asset_quantity": f"{asset_quantity:.8f}",
        }
        return self._request("POST", "/api/v1/crypto/trading/orders/", payload)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/crypto/trading/orders/{order_id}/cancel/")
