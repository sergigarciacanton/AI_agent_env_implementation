import threading
import time
import pika
import json
import ctypes
import socket
import platform
from CAV import CAV


class Environment:
    # Initialize any parameters or variables needed for the environment
    def __init__(self, logger, general):
        self.fec_list = dict()
        self.vnf_list = dict()
        self.cav = None
        self.state_changed = False
        self.logger = logger
        self.general = general
        self.rabbit_conn = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.general['control_ip'], port=self.general['rabbit_port'],
                                      credentials=pika.PlainCredentials(self.general['control_username'],
                                                                        self.general['control_password'])))
        self.subscribe_thread = threading.Thread(target=self.subscribe, args=(self.rabbit_conn, 'fec vnf'))
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()
        self.cav_thread = None

    def subscribe(self, conn, key_string):
        channel = conn.channel()

        channel.exchange_declare(exchange=self.general['control_exchange_name'], exchange_type='direct')

        queue = channel.queue_declare(queue='', exclusive=True).method.queue

        keys = key_string.split(' ')
        for key in keys:
            channel.queue_bind(
                exchange=self.general['control_exchange_name'], queue=queue, routing_key=key)

        self.logger.info('[I] Waiting for published data...')

        def callback(ch, method, properties, body):
            self.logger.debug("[D] Received. Key: " + str(method.routing_key) + ". Message: " + body.decode("utf-8"))
            if str(method.routing_key) == 'fec':
                self.fec_list = json.loads(body.decode('utf-8'))
            elif str(method.routing_key) == 'vnf':
                self.vnf_list = json.loads(body.decode('utf-8'))
                self.state_changed = True

        channel.basic_consume(
            queue=queue, on_message_callback=callback, auto_ack=True)

        channel.start_consuming()

    def start_cav(self):
        self.cav = CAV(platform.system(), self.general, self.logger)

    def reset(self):
        # Reset the environment to its initial state
        # Return the initial observation
        self.cav_thread = threading.Thread(target=self.start_cav)
        self.cav_thread.start()
        while not self.state_changed:
            time.sleep(0.001)
        self.state_changed = False
        initial_obs = dict(vnf=self.vnf_list["1"], fec_status=self.fec_list)
        info = None
        return initial_obs, info

    def step(self, action, fec_id):
        # Take an action in the environment
        # Update the state, provide a reward, and check for termination
        # Return the next observation, reward, termination flag, and additional information
        host = self.fec_list["0"]['ip']  # CAMBIAR POR fec_id!!!
        port = int(self.general['agent_fec_port'])

        fec_socket = socket.socket()
        fec_socket.connect((host, port))

        fec_socket.send(json.dumps(dict(type="action", action=action)).encode())
        response = json.loads(fec_socket.recv(1024).decode())
        if response['res'] == 200:
            self.logger.info('[I] Action ' + str(action) + ' sent successfully to FEC ' + str(fec_id))
        else:
            self.logger.critical('[!] Error from FEC' + str(response['res']))
            raise Exception
        fec_socket.close()
        while not self.state_changed:
            time.sleep(0.001)
        self.state_changed = False
        if len(self.vnf_list) > 0:
            next_obs = dict(vnf=self.vnf_list["1"], fec_status=self.fec_list)
            reward = 0
            terminated = False
        else:
            next_obs = None
            reward = 0
            terminated = True
            self.cav_thread.join()
        info = None
        return next_obs, reward, terminated, info

    def stop(self):
        killed_threads = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self.subscribe_thread.ident),
                                                                    ctypes.py_object(SystemExit))
        if killed_threads == 0:
            raise ValueError("Thread ID " + str(self.subscribe_thread.ident) + " does not exist!")
        elif killed_threads > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(self.subscribe_thread.ident, 0)
        self.logger.debug('[D] Successfully killed thread ' + str(self.subscribe_thread.ident))
        self.subscribe_thread.join()
