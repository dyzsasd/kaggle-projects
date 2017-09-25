from concurrent import futures
from io import BytesIO
import time

from PIL import Image


executor = futures.ThreadPoolExecutor(64)


def bytes_to_image(_bytes):
    return Image.open(BytesIO(_bytes))


def parse_bson_obj(obj):
    obj['imgs'] =  [
        bytes_to_image(img['picture'])
        for img in obj['imgs']
    ]
    return obj


def execute(func, argvs, sync=True):
    handles = []
    for argv in argvs:
        handles.append(executor.submit(
            func, *argv))

    if not sync:
        return handles

    is_running = True
    while is_running:
        is_running = any((it.running() for it in handles))
        finished = len([it for it in handles if it.done()])
        print("running ... %s/%s finished" % (finished, len(handles)))
        if is_running:
            time.sleep(3)

    res = {
        "total": len(handles),
        "errors": 0,
        "exceptions": []
    }

    responses = []

    for _future in handles:
        try:
            responses.append(_future.result())
        except Exception as err:
            print('error in push:')
            print(err)
            res["errors"] += 1
            res["exceptions"].append(_future.exception())
            responses.append(_future.exception())

    return responses, res


