import time
import json
import hmac
import base64
import hashlib
import requests
import uuid
from requests.exceptions import HTTPError
import concurrent.futures
import math
from json import JSONDecodeError


class WebService(object):
    def __init__(self, url, app_id, sign_key, flow_id):
        self.method = "POST"
        self.path = "/service"
        self.url = url
        self.sign_de = sign_key
        self.app_id = app_id
        self.bid = "rcm"
        self.flow_id = flow_id

    def construct_request_content(self, text_list):
        content = {"content": []}
        for text in text_list:
            content["content"].append({"text": text})
        return content

    def multi_thread_pangu_api(self, data, max_workers=3):
        datas = []
        chunk_size = math.ceil(len(data) / max_workers)
        for i in range(max_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            if start >= len(data):
                continue
            datas.append(self.construct_request_content(data[start:end]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.run_request, data): data for data in datas}
            results = []
            for futures in concurrent.futures.as_completed(futures):
                result = futures.result()
                results.extend(result)
        return results

    def run_request(self, data, wait_time=100):
        # 输入data，返回content
        req = self._request_construct(data)
        req_data = json.dumps(req)
        timestamp, signature = self._calc_sign(req_data)
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'CLOUDSOA-HMAC-SHA256 appid={}, timestamp={}, signature="{}"'.format
            (self.app_id, timestamp, signature)
        }
        try:
            response = requests.request(self.method, self.url, headers=headers, data=req_data, timeout=wait_time)
        except HTTPError as e:
            print("sit1: request is failure. ", e)
            return None
        except requests.RequestException as e:
            print("sit2: request is failure. ", e)
            return None
        except Exception as e:
            print("sit3: request is failure. ", e)
            return None
        response = self._response_parse(response)
        return response

    def _request_construct(self, data):
        msg_uuid = str(uuid.uuid1())
        request_data = {
            "data": data,
            "meta": {
                "bId": self.bid,
                "flowId": self.flow_id,
                "uuId": msg_uuid
            },
            "version": "1.0"
        }
        return request_data

    def _calc_sign(self, data):
        timestamp = str(int(time.time() * 1000))
        items = [self.method, self.path, "", data, f'appid={self.app_id}', f'timestamp={timestamp}']
        sign_str = '&'.join(items)
        sign_str_encode = sign_str.encode('utf-8')
        sign_key = self.sign_de.encode('utf-8')
        signature = base64.b64encode(hmac.new(sign_key, sign_str_encode, digestmod=hashlib.sha256).digest()).decode(
            "utf-8")
        return timestamp, signature

    @staticmethod
    def _response_parse(response):
        try:
            res = response.json()
        except JSONDecodeError:
            try:
                res = json.loads(response.text)
            except JSONDecodeError:
                res = None
            except Exception as e:
                res = None
                print("sit4: parse request result failure !", e)

        if res:
            res_content = res.get("result", {}).get("content", {})
            if res_content:
                return res_content
            else:
                print("sit5: parse request result failure !", res)
                return None
        return None
