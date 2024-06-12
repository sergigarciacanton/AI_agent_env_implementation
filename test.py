from prometheus_client import start_http_server
import os
import multiprocessing
import time


def worker(stop):
    server, t = start_http_server(addr='10.0.0.11', port=8000)
    while not stop.value:
        time.sleep(0.001)
    server.shutdown()
    t.join()

stop = multiprocessing.Value('b', False)
process = multiprocessing.Process(target=worker, args=(stop,))
process.start()

input('Press...')

stop.value = True

input('Press...')

#server2, t2 = start_http_server(addr='10.0.0.11', port=8000)

#input('Press...')

#server2.shutdown()
#t2.join()
