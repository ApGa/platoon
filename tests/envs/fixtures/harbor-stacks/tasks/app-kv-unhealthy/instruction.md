A key-value store runs at `$KV_URL` (`http://kv:6379`): `PUT $KV_URL/<key>` stores the request's body under `<key>`,
and `GET $KV_URL/<key>` answers it. Store the value `forty-two` under the key `answer`.
