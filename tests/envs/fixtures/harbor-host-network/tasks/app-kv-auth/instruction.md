A key-value store runs at `$KV_URL`: `PUT $KV_URL/<key>` stores the request's body under `<key>`, and
`GET $KV_URL/<key>` answers it. Where `$KV_TOKEN` is set, the store answers only requests that carry it
(`Authorization: Bearer $KV_TOKEN`). Store the value `forty-two` under the key `answer`.
