# RH PROPR Quant Engine — Live/Paper Trading Workstation

This package upgrades the earlier research dashboard into a more fully wired desktop workstation:

- functional page navigation for Dashboard, Live Trading, Portfolio, Logs, and Config
- paper-trading simulation with positions, equity, risk table, and trade log
- credential persistence to `.runtime_settings.json`
- Robinhood live-account connectivity scaffold
- manual market/limit order entry and cancel-request UI
- log export and runtime settings save/load

## Important

This package is **not certified production-ready**. It is a hardened operator workstation and integration scaffold.
Before trading real funds, you should still complete:

- Robinhood API schema reconciliation against the latest official docs
- exchange/order-type validation in a test account
- persistent audit logging and fill reconciliation
- unit/integration tests and fault-injection tests
- secure secrets handling (OS keychain or HSM, not plain local JSON)

## Run on Windows CMD

```cmd
python -m pip install -r requirements.txt
python app.py
```

## Generate a preview image

```cmd
set QT_QPA_PLATFORM=offscreen
python render_preview.py
```

## Live Trading page

Paste your Robinhood API key and Ed25519 private key into the header fields, then:

1. Open **Live Trading**
2. Click **Connect & Refresh Account**
3. Review the account snapshot
4. Submit manual orders only after confirming the API response schema in the official docs
