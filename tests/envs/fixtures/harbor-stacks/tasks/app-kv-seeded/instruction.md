A key-value store runs at `$KV_URL` (`http://kv:6379`): `PUT $KV_URL/<key>` stores the request's body under `<key>`,
and `GET $KV_URL/<key>` answers it. Store the value it holds under the key `greeting` under the key `answer` too.
