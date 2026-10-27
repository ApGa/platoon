An order API runs at `$API_URL` (`http://api:8000`). Place exactly three orders, with three different items: each
order is a `POST` to `$API_URL/orders` whose body is a JSON object with an `item`, such as `{"item": "apple"}`.
